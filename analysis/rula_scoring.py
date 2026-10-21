"""RULA scoring based on the supplied lecture slides and scoring tables.

Scores one observed side and one selected posture at a time.
Posture categories and task conditions must be assessed and entered;
they are not inferred automatically from camera angles.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict


# Table A.
# Indexed as:
#   upper-arm score 1..6
#   lower-arm score 1..3
#   wrist score 1..4 and wrist-twist score 1..2
#
# Each row has eight entries in this order:
#   wrist 1 / twist 1, wrist 1 / twist 2,
#   wrist 2 / twist 1, wrist 2 / twist 2,
#   wrist 3 / twist 1, wrist 3 / twist 2,
#   wrist 4 / twist 1, wrist 4 / twist 2.
_TABLE_A = (
    (
        (1, 2, 2, 2, 2, 3, 3, 3),
        (2, 2, 2, 2, 3, 3, 3, 3),
        (2, 3, 2, 3, 3, 3, 4, 4),
    ),
    (
        (2, 2, 2, 3, 3, 3, 4, 4),
        (2, 2, 2, 3, 3, 3, 4, 4),
        (2, 3, 3, 3, 3, 4, 4, 5),
    ),
    (
        (2, 3, 3, 3, 4, 4, 5, 6),
        (3, 4, 4, 4, 4, 4, 5, 5),
        (4, 4, 4, 4, 4, 5, 5, 5),
    ),
    (
        (4, 4, 4, 4, 4, 5, 5, 5),
        (4, 4, 4, 4, 4, 5, 5, 5),
        (4, 4, 4, 5, 5, 5, 6, 6),
    ),
    (
        (5, 5, 5, 5, 5, 6, 6, 7),
        (5, 6, 6, 6, 6, 6, 7, 7),
        (6, 6, 6, 7, 7, 7, 7, 8),
    ),
    (
        (7, 7, 7, 7, 7, 8, 8, 9),
        (8, 8, 8, 8, 8, 9, 9, 9),
        (9, 9, 9, 9, 9, 9, 9, 9),
    ),
)


# Table B.
# Indexed by neck score 1..6, then by trunk score 1..6 and leg score 1..2.
# Each row has twelve entries:
#   trunk 1 / legs 1, trunk 1 / legs 2, ..., trunk 6 / legs 2.
_TABLE_B = (
    (1, 3, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7),
    (2, 3, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7),
    (3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7),
    (5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8),
    (7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8),
    (8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9),
)


# Table C.
# Rows are final Wrist/Arm scores 1..8+.
# Columns are final Neck/Trunk/Leg scores 1..7+.
_TABLE_C = (
    (1, 2, 3, 3, 4, 5, 5),
    (2, 2, 3, 4, 4, 5, 5),
    (3, 3, 3, 4, 4, 5, 6),
    (3, 3, 3, 4, 5, 6, 6),
    (4, 4, 4, 5, 6, 7, 7),
    (4, 4, 5, 6, 6, 7, 7),
    (5, 5, 6, 6, 7, 7, 7),
    (5, 5, 6, 7, 7, 7, 7),
)


@dataclass
class RULAObservation:
    """One side's posture and task observations for the RULA worksheet."""

    # Step 1: choose the base upper-arm posture score from the slides (1..4).
    upper_arm_score: int

    # Step 2: choose the base lower-arm posture score from the slides (1..2).
    lower_arm_score: int

    # Step 3: wrist posture score before the deviation adjustment (1..3).
    wrist_score: int

    # Step 4: wrist twist score: 1 = mid-range, 2 = near end range.
    wrist_twist_score: int

    # Step 9: base neck posture score before adjustments (1..4).
    neck_score: int

    # Step 10: base trunk posture score before adjustments (1..4).
    trunk_score: int

    # Step 11: True if legs and feet are supported; otherwise score is 2.
    legs_and_feet_supported: bool

    # Step 1a adjustments.
    shoulder_raised: bool = False
    upper_arm_abducted: bool = False
    arm_supported_or_person_leaning: bool = False

    # Step 2a: add 1 if the arm works across the midline or out to the side.
    arm_across_midline_or_out_to_side: bool = False

    # Step 3a adjustment.
    wrist_bent_from_midline: bool = False

    # Steps 9a and 10a adjustments.
    neck_twisted: bool = False
    neck_side_bent: bool = False
    trunk_twisted: bool = False
    trunk_side_bent: bool = False

    # Step 6 / Step 13: supplied lecture slides say >1 minute OR >4 actions/min.
    held_static_minutes: float = 0.0
    repeated_actions_per_minute: float = 0.0

    # Step 7 / Step 14. Weight is entered in pounds, as in the slides.
    load_lbs: float = 0.0
    load_mode: str = "intermittent"  # "intermittent", "static", or "repeated"
    shock_loading: bool = False

    # RULA evaluates one side at a time.
    side: str = "right"
    posture_note: str = ""


@dataclass
class RULAResult:
    """Detailed RULA output, including intermediate scores and final rating."""

    side: str
    adjusted_component_scores: Dict[str, int]
    muscle_use_score: int
    force_load_score: int
    posture_score_a: int
    posture_score_b: int
    final_wrist_arm_score: int
    final_neck_trunk_leg_score: int
    final_rula_score: int
    action_level: int
    risk_level: str
    rating: str
    action: str
    posture_note: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return asdict(self)


def _require_int(name: str, value: int, minimum: int, maximum: int) -> None:
    if not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be an integer from {minimum} to {maximum}; got {value!r}"
        )


def _force_load_score(observation: RULAObservation) -> int:
    """Apply the supplied slides' Step 7 / Step 14 load categories."""
    if observation.load_lbs < 0:
        raise ValueError("load_lbs cannot be negative")

    if observation.load_mode not in {"intermittent", "static", "repeated"}:
        raise ValueError(
            "load_mode must be 'intermittent', 'static', or 'repeated'"
        )

    # The lecture slides assign the highest category to >20 lb static/repeated
    # loads, or to shock / rapidly building forces.
    if observation.shock_loading:
        return 3

    if observation.load_lbs > 20.0:
        if observation.load_mode in {"static", "repeated"}:
            return 3
        raise ValueError(
            "The supplied slides do not specify a score for an intermittent "
            "load over 20 lb. Review this case against the RULA worksheet."
        )

    if 5.0 <= observation.load_lbs <= 20.0:
        if observation.load_mode == "intermittent":
            return 1
        return 2  # Static or repeated load in the slides' 5–20 lb band.

    if observation.load_lbs < 5.0 and observation.load_mode == "intermittent":
        return 0

    # The slides do not state a category for <5 lb held statically or repeated.
    raise ValueError(
        "The supplied slides do not specify a load score for loads under "
        "5 lb in static or repeated use. Review that case against the worksheet."
    )


def score_rula(observation: RULAObservation) -> RULAResult:
    """Calculate one posture's RULA score using the supplied scoring tables."""

    if observation.side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    _require_int("upper_arm_score", observation.upper_arm_score, 1, 4)
    _require_int("lower_arm_score", observation.lower_arm_score, 1, 2)
    _require_int("wrist_score", observation.wrist_score, 1, 3)
    _require_int("wrist_twist_score", observation.wrist_twist_score, 1, 2)
    _require_int("neck_score", observation.neck_score, 1, 4)
    _require_int("trunk_score", observation.trunk_score, 1, 4)

    if observation.held_static_minutes < 0:
        raise ValueError("held_static_minutes cannot be negative")
    if observation.repeated_actions_per_minute < 0:
        raise ValueError("repeated_actions_per_minute cannot be negative")

    # Steps 1 and 1a: upper arm, including the slide's adjustments.
    upper_arm = observation.upper_arm_score
    upper_arm += int(observation.shoulder_raised)
    upper_arm += int(observation.upper_arm_abducted)
    upper_arm -= int(observation.arm_supported_or_person_leaning)
    upper_arm = max(1, min(6, upper_arm))

    # Steps 2 and 2a: add one point if either condition applies.
    lower_arm = observation.lower_arm_score
    lower_arm += int(observation.arm_across_midline_or_out_to_side)
    lower_arm = max(1, min(3, lower_arm))

    # Steps 3 and 3a.
    wrist = observation.wrist_score
    wrist += int(observation.wrist_bent_from_midline)
    wrist = max(1, min(4, wrist))

    # Step 4: validate the selected wrist-twist category.
    wrist_twist = observation.wrist_twist_score

    # Step 5: Table A lookup.
    table_a_row = _TABLE_A[upper_arm - 1][lower_arm - 1]
    table_a_index = (wrist - 1) * 2 + (wrist_twist - 1)
    posture_score_a = table_a_row[table_a_index]

    # Steps 9 and 9a: neck posture and adjustments.
    neck = observation.neck_score
    neck += int(observation.neck_twisted)
    neck += int(observation.neck_side_bent)
    neck = max(1, min(6, neck))

    # Steps 10 and 10a: trunk posture and adjustments.
    trunk = observation.trunk_score
    trunk += int(observation.trunk_twisted)
    trunk += int(observation.trunk_side_bent)
    trunk = max(1, min(6, trunk))

    # Step 11: supported legs and feet score 1; otherwise score 2.
    legs = 1 if observation.legs_and_feet_supported else 2

    # Step 12: Table B lookup.
    table_b_index = (trunk - 1) * 2 + (legs - 1)
    posture_score_b = _TABLE_B[neck - 1][table_b_index]

    # Steps 6 and 13: use the thresholds stated in the supplied slides.
    muscle_use_score = int(
        observation.held_static_minutes > 1.0
        or observation.repeated_actions_per_minute > 4.0
    )

    # Steps 7 and 14.
    force_load_score = _force_load_score(observation)

    # Steps 8 and 15: add muscle-use and force/load points to each group.
    final_wrist_arm_score = (
        posture_score_a + muscle_use_score + force_load_score
    )
    final_neck_trunk_leg_score = (
        posture_score_b + muscle_use_score + force_load_score
    )

    # Table C uses row 8+ and column 7+ for values above those limits.
    table_c_row = min(final_wrist_arm_score, 8) - 1
    table_c_column = min(final_neck_trunk_leg_score, 7) - 1
    final_rula_score = _TABLE_C[table_c_row][table_c_column]

    # Risk levels and actions follow the supplied RULA tables.
    if final_rula_score <= 2:
        action_level = 1
        risk_level = "Negligible"
        rating = "Acceptable"
        action = (
            "Acceptable if the posture is not maintained or repeated for long periods."
        )
    elif final_rula_score <= 4:
        action_level = 2
        risk_level = "Low"
        rating = "Investigate further"
        action = "Investigate further."
    elif final_rula_score <= 6:
        action_level = 3
        risk_level = "Medium"
        rating = "Investigate further and change soon"
        action = "Investigate further and change soon."
    else:
        action_level = 4
        risk_level = "High"
        rating = "Investigate and change immediately"
        action = "Investigate and change immediately."

    return RULAResult(
        side=observation.side,
        adjusted_component_scores={
            "upper_arm": upper_arm,
            "lower_arm": lower_arm,
            "wrist": wrist,
            "wrist_twist": wrist_twist,
            "neck": neck,
            "trunk": trunk,
            "legs": legs,
        },
        muscle_use_score=muscle_use_score,
        force_load_score=force_load_score,
        posture_score_a=posture_score_a,
        posture_score_b=posture_score_b,
        final_wrist_arm_score=final_wrist_arm_score,
        final_neck_trunk_leg_score=final_neck_trunk_leg_score,
        final_rula_score=final_rula_score,
        action_level=action_level,
        risk_level=risk_level,
        rating=rating,
        action=action,
        posture_note=observation.posture_note,
    )
