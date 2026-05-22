import math
import torch
import torch.nn as nn
import escnn.nn as enn
from escnn import gspaces
from models.utils import FourierPointwiseInnerBn
import time


def _t(label, t0):
    print(f"    [{time.time() - t0:.2f}s] {label}")


BN_MAP_2d = {
    "IIDbn": enn.IIDBatchNorm2d,
    "Normbn": enn.NormBatchNorm,
    "GNormBatchNorm": enn.GNormBatchNorm
}


class _EquivariantELU(enn.EquivariantModule):
    """Pointwise ELU, equivariant for representations acting by permutation (e.g. regular_repr)."""
    def __init__(self, in_type: enn.FieldType):
        super().__init__()
        self.in_type = in_type
        self.out_type = in_type
        self._elu = torch.nn.ELU(inplace=False)

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        return enn.GeometricTensor(self._elu(x.tensor), self.out_type)

    def evaluate_output_shape(self, input_shape):
        return input_shape


class EquiUNet(torch.nn.Module):
    def __init__(self,
                 in_channels=1,
                 n_classes=1,
                 depth: int = 4,
                 start_filters: int = 64,
                 max_rot_order=2,
                 group_order=-1,
                 activation_type="gated",
                 fourier_N=8,
                 fourier_function="p_relu",
                 bn_type="IIDbn",
                 conv_sigma=0.6,
                 kernel_size=7):
        super().__init__()
        b = start_filters
        t0 = time.time()

        print(f"EquiUNet: in={in_channels}, out={n_classes}, depth={depth}, "
              f"start_filters={b}")

        if group_order == -1:
            self.r2_act = gspaces.rot2dOnR2(N=-1, maximum_frequency=max_rot_order)
        else:
            self.r2_act = gspaces.rot2dOnR2(N=group_order)
        print(f"  Group: {self.r2_act.name} (N={group_order}), activation: {activation_type}")

        self.group_order = group_order
        self.max_rot_order = max_rot_order
        self.activation_type = activation_type
        self.fourier_N = fourier_N
        self.fourier_function = fourier_function
        self.bn_type = bn_type
        self.conv_sigma = conv_sigma
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.batch_norm = self._create_bn()

        self.input_type = enn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])

        # Scale start_filters so that EquiUNet(start_filters=N) ≈ UNet(start_filters=N) in params.
        # Equivariant conv params ∝ basis_dim × C², standard conv params ∝ k² × C².
        # Equal params requires C_equiv = N × sqrt(k² / basis_dim_per_pair).
        scale = self._channel_scale()
        # Round per-level so that deeper layers benefit from independent rounding.
        channels = [max(1, round(start_filters * (2 ** i) * scale)) for i in range(depth + 1)]
        print(f"  Channel scale: {scale:.3f} → channels: {channels} (from start_filters={start_filters})")

        # --- Stem ---
        print(f"  Building encoder (depth={depth})...")
        self.initial_conv = self._build_double_conv(self.input_type, channels[0])
        _t(f"stem: {in_channels} -> {channels[0]}", t0)

        # --- Encoder ---
        self.pools = nn.ModuleList()
        self.down_convs = nn.ModuleList()

        skip_types = []
        cur_type = self.initial_conv.out_type
        skip_types.append(cur_type)

        for i in range(depth):
            out_ch = channels[i + 1]
            pool = self._build_pool(cur_type)
            conv = self._build_double_conv(cur_type, out_ch)
            self.pools.append(pool)
            self.down_convs.append(conv)
            cur_type = conv.out_type
            _t(f"down{i}: {channels[i]} -> {out_ch}", t0)
            if i < depth - 1:
                skip_types.append(cur_type)

        # --- Decoder ---
        print(f"  Building decoder...")
        self.up_upsamples = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for i in range(depth):
            skip_type = skip_types[depth - 1 - i]
            out_ch = channels[depth - 1 - i]
            upsample = self._build_upsample(cur_type)
            conv = self._build_double_conv(cur_type + skip_type, out_ch)
            self.up_upsamples.append(upsample)
            self.up_convs.append(conv)
            _t(f"up{i}: {cur_type.size}+{skip_type.size} -> {out_ch}", t0)
            cur_type = conv.out_type

        # --- Output ---
        out_type = enn.FieldType(self.r2_act, n_classes * [self.r2_act.trivial_repr])
        self.out_conv = enn.R2Conv(cur_type, out_type, kernel_size=1, padding=0,
                                   sigma=self.conv_sigma, bias=True)
        print(f"  EquiUNet initialized.")

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)

        # Encoder — collect skip tensors
        cur = self.initial_conv(x)
        skips = [cur]
        for pool, down_conv in zip(self.pools, self.down_convs):
            cur = down_conv(pool(cur))
            skips.append(cur)

        # skips[-1] is the bottleneck; skips[:-1] are the skip connections
        for upsample, up_conv, skip in zip(self.up_upsamples, self.up_convs, reversed(skips[:-1])):
            cur = up_conv(enn.tensor_directsum([upsample(cur), skip]))

        return self.out_conv(cur).tensor

    # --- Building blocks ---

    def _channel_scale(self) -> float:
        """
        Compute sqrt(k² / basis_dim_per_channel_pair) so that multiplying start_filters
        by this factor gives an equivariant channel count with the same parameter budget
        as a standard UNet with that start_filters value.
        """
        if self.activation_type == "fourierbn":
            G = self.r2_act.fibergroup
            irreps = G.bl_irreps(self.max_rot_order)
            act = FourierPointwiseInnerBn(self.r2_act, channels=1, irreps=irreps,
                                          function=self.fourier_function, N=self.fourier_N)
            ref_in = act.in_type
            ref_out = act.in_type
        elif self.activation_type == "regular":
            ref_in = ref_out = enn.FieldType(self.r2_act, [self.r2_act.regular_repr])
        else:  # gated: dominant conv is activated_type → full_field (gates included in out)
            scalar_field, vector_field, _, full_field = self._build_field_type(1)
            ref_in  = scalar_field + vector_field  # activated_type
            ref_out = full_field

        tmp = enn.R2Conv(ref_in, ref_out, kernel_size=self.kernel_size, sigma=0.6, bias=False)
        basis_dim = float(tmp.basisexpansion.dimension())
        del tmp
        return math.sqrt(3 ** 2 / basis_dim)

    def _build_double_conv(self, in_type, out_channels):
        if self.activation_type == "fourierbn":
            return self._build_double_conv_fourierbn(in_type, out_channels)
        if self.activation_type == "regular":
            return self._build_double_conv_regular(in_type, out_channels)
        return self._build_double_conv_gated(in_type, out_channels)

    def _build_double_conv_regular(self, in_type, out_channels):
        """R2Conv -> InnerBN -> ELU (x2), using regular representation. For discrete groups."""
        out_type = enn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])

        conv1 = enn.R2Conv(in_type, out_type, kernel_size=self.kernel_size, padding=self.padding,
                           sigma=self.conv_sigma, bias=True)
        bn1 = enn.InnerBatchNorm(out_type)
        nonlin1 = _EquivariantELU(out_type)

        conv2 = enn.R2Conv(out_type, out_type, kernel_size=self.kernel_size, padding=self.padding,
                           sigma=self.conv_sigma, bias=True)
        bn2 = enn.InnerBatchNorm(out_type)
        nonlin2 = _EquivariantELU(out_type)

        return enn.SequentialModule(conv1, bn1, nonlin1, conv2, bn2, nonlin2)

    def _build_double_conv_fourierbn(self, in_type, out_channels):
        """R2Conv -> FourierPointwiseInnerBn (x2), using spectral regular representation."""
        G = self.r2_act.fibergroup
        irreps = G.bl_irreps(self.max_rot_order)

        act1 = FourierPointwiseInnerBn(self.r2_act, channels=out_channels, irreps=irreps,
                                       function=self.fourier_function, N=self.fourier_N)
        feat_type = act1.in_type

        conv1 = enn.R2Conv(in_type, feat_type, kernel_size=self.kernel_size, padding=self.padding,
                           sigma=self.conv_sigma, bias=True)

        act2 = FourierPointwiseInnerBn(self.r2_act, channels=out_channels, irreps=irreps,
                                       function=self.fourier_function, N=self.fourier_N)
        conv2 = enn.R2Conv(feat_type, feat_type, kernel_size=self.kernel_size, padding=self.padding,
                           sigma=self.conv_sigma, bias=True)

        return enn.SequentialModule(conv1, act1, conv2, act2)

    def _build_double_conv_gated(self, in_type, out_channels):
        """R2Conv -> BN -> GatedNonlin (x2), using scalar+vector fields. For continuous SO(2)."""
        scalar_field, vector_field, gate_field, full_field = self._build_field_type(out_channels)
        activated_type = scalar_field + vector_field

        conv1 = enn.R2Conv(in_type, full_field, kernel_size=self.kernel_size, padding=self.padding,
                           sigma=self.conv_sigma, bias=True)
        bn1 = self._build_batch_norm(full_field)
        nonlin1 = self._build_gated_nonlinearity(scalar_field, gate_field, full_field)

        conv2 = enn.R2Conv(activated_type, full_field, kernel_size=self.kernel_size, padding=self.padding,
                           sigma=self.conv_sigma, bias=True)
        bn2 = self._build_batch_norm(full_field)
        nonlin2 = self._build_gated_nonlinearity(scalar_field, gate_field, full_field)

        return enn.SequentialModule(conv1, bn1, nonlin1, conv2, bn2, nonlin2)

    def _build_field_type(self, channels):
        vector_rep = []
        for irr in self.r2_act.irreps:
            if irr.is_trivial():
                continue
            mult = int(irr.size // irr.sum_of_squares_constituents)
            vector_rep.extend([irr] * mult * channels)

        scalar_rep = [self.r2_act.trivial_repr] * channels
        scalar_field = enn.FieldType(self.r2_act, scalar_rep)
        vector_field = enn.FieldType(self.r2_act, vector_rep)
        gate_repr = [self.r2_act.trivial_repr] * len(vector_rep)
        gate_field = enn.FieldType(self.r2_act, gate_repr) + vector_field
        full_field = scalar_field + gate_field
        return scalar_field, vector_field, gate_field, full_field

    def _build_gated_nonlinearity(self, scalar_field, gate_field, full_field):
        return enn.MultipleModule(
            in_type=full_field,
            labels=['scalar'] * len(scalar_field) + ['gated'] * len(gate_field),
            modules=[
                (enn.ELU(scalar_field), 'scalar'),
                (enn.GatedNonLinearity1(gate_field), 'gated')
            ],
            reshuffle=0
        )

    def _build_pool(self, in_type):
        labels = ["trivial" if r.is_trivial() else "others" for r in in_type]
        cur_type_labeled = in_type.group_by_labels(labels)

        has_trivial = "trivial" in cur_type_labeled
        has_others = "others" in cur_type_labeled

        if not has_trivial:
            return enn.NormMaxPool(in_type, kernel_size=2)
        if not has_others:
            return enn.PointwiseMaxPool2D(in_type, kernel_size=2)

        trivials = cur_type_labeled["trivial"]
        others = cur_type_labeled["others"]
        return enn.MultipleModule(
            in_type=in_type,
            labels=labels,
            modules=[
                (enn.PointwiseMaxPool2D(trivials, kernel_size=2), 'trivial'),
                (enn.NormMaxPool(others, kernel_size=2), 'others')
            ],
            reshuffle=0
        )

    def _build_upsample(self, in_type):
        return enn.R2Upsampling(in_type, scale_factor=2)

    def _build_batch_norm(self, in_type):
        labels = ["trivial" if r.is_trivial() else "others" for r in in_type]
        cur_type_labeled = in_type.group_by_labels(labels)
        trivials = cur_type_labeled["trivial"]
        others = cur_type_labeled["others"]
        if len(others) == 0:
            return enn.InnerBatchNorm(in_type)
        return enn.MultipleModule(
            in_type=in_type,
            labels=labels,
            modules=[
                (enn.InnerBatchNorm(trivials), 'trivial'),
                (self.batch_norm(others), 'others')
            ],
            reshuffle=0
        )

    def _create_bn(self):
        try:
            return BN_MAP_2d[self.bn_type]
        except KeyError:
            raise ValueError(f"Unsupported batch norm type: {self.bn_type}")
