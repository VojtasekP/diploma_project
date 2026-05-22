from dataclasses import dataclass


@dataclass(frozen=True)
class LinkingParameters:
    birth_death_cost: float
    edge_removal_cost: float
    feature_cost_multiplier: float
    maximum_distance: float
    maximum_skipped_frames: int = 1
    minimum_length: int = 5
