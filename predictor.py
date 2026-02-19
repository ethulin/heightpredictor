"""
Adult Height Predictor based on Cole & Wright (2011)
"A chart to predict adult height from a child's current height"
Annals of Human Biology, 38(6), 662-668.

Algorithm:
  1. Convert child's height to z-score using CDC LMS growth reference
  2. Apply regression to the mean: z_adult = r(age, sex) * z_child
     where r is the child-adult height correlation from the Zurich Growth Study
  3. Convert predicted adult z-score back to height
  4. Confidence interval: SE = SD_adult * sqrt(1 - r^2)
"""

import csv
import os
import math
from scipy.interpolate import interp1d
import numpy as np

# ---------------------------------------------------------------------------
# Child-adult height correlations digitized from Figure 1 of Cole & Wright 2011
# Source: First Zurich Longitudinal Growth Study (Molinari et al. 1995)
#
# Anchored values from the paper text:
#   - Birth (~0.4 for both sexes)
#   - Boys age 2: 0.75, Girls age 2: 0.65
#   - Boys age 3: 0.764, Girls age 3: 0.660  (from worked example, p.664)
#   - Girls age 9: 0.72  (from Discussion, p.667)
#   - Boys age 11: ~0.85 (puberty onset, Discussion p.667)
#   - Both sexes approach 1.0 at age 20
# Remaining values interpolated from Figure 1 graphical data.
# ---------------------------------------------------------------------------

# fmt: off
CORRELATION_BOYS_AGE =  [ 0,   0.5,  1,    1.5,  2,    3,     4,    5,    6,    7,    8,    9,    10,   11,   12,   13,   14,   15,   16,   17,   18,   19,   20]
CORRELATION_BOYS_R =    [0.38, 0.50, 0.60, 0.68, 0.75, 0.764, 0.79, 0.81, 0.83, 0.84, 0.855,0.865,0.875,0.88, 0.85, 0.80, 0.82, 0.88, 0.92, 0.96, 0.98, 0.99, 1.0]

CORRELATION_GIRLS_AGE = [ 0,   0.5,  1,    1.5,  2,    3,     4,    5,    6,    7,    8,    9,    10,   11,   12,   13,   14,   15,   16,   17,   18,   19,   20]
CORRELATION_GIRLS_R =   [0.38, 0.44, 0.52, 0.58, 0.65, 0.660, 0.68, 0.70, 0.71, 0.72, 0.72, 0.72, 0.68, 0.62, 0.65, 0.75, 0.84, 0.90, 0.95, 0.97, 0.99, 0.995,1.0]
# fmt: on

_interp_boys = interp1d(CORRELATION_BOYS_AGE, CORRELATION_BOYS_R, kind="cubic", fill_value="extrapolate")
_interp_girls = interp1d(CORRELATION_GIRLS_AGE, CORRELATION_GIRLS_R, kind="cubic", fill_value="extrapolate")


def get_correlation(age_years: float, sex: str) -> float:
    """Get the child-adult height correlation for a given age and sex.

    Args:
        age_years: Age in years (0-20)
        sex: 'M' or 'F'

    Returns:
        Correlation coefficient r
    """
    age_years = max(0, min(20, age_years))
    if sex.upper() == "M":
        r = float(_interp_boys(age_years))
    else:
        r = float(_interp_girls(age_years))
    return max(0, min(1.0, r))


# ---------------------------------------------------------------------------
# CDC LMS data for length/stature-for-age
#   - lenageinf.csv: recumbent length, birth to 36 months
#   - statage.csv: standing stature, 24 to 240 months (2-20 years)
# For ages 0-24 months we use infant length data; for 24+ months stature data.
# ---------------------------------------------------------------------------

_lms_infant = {"M": [], "F": []}
_lms_stature = {"M": [], "F": []}
_lms_loaded = False


def _load_lms():
    global _lms_loaded
    if _lms_loaded:
        return
    base = os.path.dirname(__file__)

    # Infant length-for-age (birth to 36 months)
    infant_path = os.path.join(base, "lenageinf.csv")
    with open(infant_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sex_code = int(row["Sex"])
            sex = "M" if sex_code == 1 else "F"
            _lms_infant[sex].append({
                "age_months": float(row["Agemos"]),
                "L": float(row["L"]),
                "M": float(row["M"]),
                "S": float(row["S"]),
            })

    # Stature-for-age (24 to 240 months)
    stature_path = os.path.join(base, "statage.csv")
    with open(stature_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sex_code = int(row["Sex"])
            sex = "M" if sex_code == 1 else "F"
            _lms_stature[sex].append({
                "age_months": float(row["Agemos"]),
                "L": float(row["L"]),
                "M": float(row["M"]),
                "S": float(row["S"]),
            })
    _lms_loaded = True


def _get_lms(age_months: float, sex: str) -> tuple:
    """Get interpolated L, M, S values for a given age and sex.

    Uses infant length data for ages < 24 months, stature data for 24+ months.

    Returns: (L, M, S)
    """
    _load_lms()
    sex = sex.upper()

    if age_months < 24:
        data = _lms_infant[sex]
    else:
        data = _lms_stature[sex]

    ages = [d["age_months"] for d in data]
    Ls = [d["L"] for d in data]
    Ms = [d["M"] for d in data]
    Ss = [d["S"] for d in data]

    age_months = max(ages[0], min(ages[-1], age_months))

    L = float(interp1d(ages, Ls, kind="linear")(age_months))
    M = float(interp1d(ages, Ms, kind="linear")(age_months))
    S = float(interp1d(ages, Ss, kind="linear")(age_months))
    return L, M, S


def height_to_zscore(height_cm: float, age_years: float, sex: str) -> float:
    """Convert a child's height to a z-score using CDC LMS data.

    Uses the LMS method: z = ((height/M)^L - 1) / (L * S)

    For L close to 0, uses the log form: z = ln(height/M) / S

    Args:
        height_cm: Height in centimeters
        age_years: Age in years (2-20)
        sex: 'M' or 'F'

    Returns:
        z-score (SD score)
    """
    age_months = age_years * 12
    L, M, S = _get_lms(age_months, sex)

    if abs(L) < 0.001:
        z = math.log(height_cm / M) / S
    else:
        z = ((height_cm / M) ** L - 1) / (L * S)
    return z


def zscore_to_height(z: float, age_years: float, sex: str) -> float:
    """Convert a z-score back to height using CDC LMS data.

    Inverse of height_to_zscore.
    """
    age_months = age_years * 12
    L, M, S = _get_lms(age_months, sex)

    if abs(L) < 0.001:
        height = M * math.exp(S * z)
    else:
        height = M * (1 + L * S * z) ** (1 / L)
    return height


# Adult reference values at age 20 from CDC data
# Males:   M = 176.85, SD = M*S = 176.85 * 0.04037 ≈ 7.14 cm
# Females: M = 163.34, SD = M*S = 163.34 * 0.03964 ≈ 6.47 cm
#
# The paper uses British 1990: Males M=177.3, SD=7.0; Females M=163.6, SD=6.0
# We use CDC values for consistency since we use CDC for child z-scores.

ADULT_MEAN = {"M": 176.849, "F": 163.338}
ADULT_SD = {"M": 7.14, "F": 6.47}


def predict_adult_height(
    height_cm: float,
    age_years: float,
    sex: str,
    confidence: float = 0.80,
) -> dict:
    """Predict adult height from a child's current height.

    Implements Cole & Wright (2011) equations 5-7:
      z_adult = r * z_child                       (Eq. 5)
      Height_adult = SD_a * r * z_c + Mean_a      (Eq. 6)
      SE(Height_a) = SD_a * sqrt(1 - r^2)         (Eq. 7)

    Args:
        height_cm: Child's current height in cm
        age_years: Child's age in years (2-20)
        sex: 'M' for male, 'F' for female
        confidence: Confidence level for the interval (default 0.80 per paper)

    Returns:
        Dictionary with prediction results
    """
    sex = sex.upper()
    if sex not in ("M", "F"):
        raise ValueError("sex must be 'M' or 'F'")
    if age_years < 0 or age_years > 20:
        raise ValueError("age must be between 0 and 20 years")

    # Step 1: Convert child's height to z-score (Eq. 1)
    z_child = height_to_zscore(height_cm, age_years, sex)

    # Step 2: Get child-adult correlation for this age/sex
    r = get_correlation(age_years, sex)

    # Step 3: Regression to the mean (Eq. 5)
    z_adult = r * z_child

    # Step 4: Convert to predicted adult height (Eq. 6)
    mean_a = ADULT_MEAN[sex]
    sd_a = ADULT_SD[sex]
    predicted_height = sd_a * z_adult + mean_a

    # Step 5: Standard error (Eq. 7)
    se = sd_a * math.sqrt(1 - r ** 2)

    # Step 6: Confidence interval
    from scipy.stats import norm
    z_crit = norm.ppf(1 - (1 - confidence) / 2)
    ci_lower = predicted_height - z_crit * se
    ci_upper = predicted_height + z_crit * se

    # Child's percentile
    from scipy.stats import norm as norm_dist
    child_percentile = norm_dist.cdf(z_child) * 100

    return {
        "predicted_height_cm": round(predicted_height, 1),
        "predicted_height_ft": _cm_to_ft_in(predicted_height),
        "ci_lower_cm": round(ci_lower, 1),
        "ci_upper_cm": round(ci_upper, 1),
        "ci_lower_ft": _cm_to_ft_in(ci_lower),
        "ci_upper_ft": _cm_to_ft_in(ci_upper),
        "confidence_level": confidence,
        "standard_error_cm": round(se, 1),
        "child_zscore": round(z_child, 2),
        "child_percentile": round(child_percentile, 1),
        "correlation": round(r, 3),
        "input": {
            "height_cm": height_cm,
            "age_years": age_years,
            "sex": sex,
        },
    }


def _cm_to_ft_in(cm: float) -> str:
    """Convert cm to feet and inches string."""
    inches_total = cm / 2.54
    feet = int(inches_total // 12)
    inches = inches_total % 12
    return f"{feet}'{inches:.1f}\""


def predict_from_inches(height_inches: float, age_years: float, sex: str, confidence: float = 0.80) -> dict:
    """Convenience wrapper accepting height in inches."""
    return predict_adult_height(height_inches * 2.54, age_years, sex, confidence)
