#!/usr/bin/env python3
"""
Adult Height Predictor CLI

Predicts adult height from a child's current height, age, and sex.
Based on Cole & Wright (2011) regression-to-the-mean method.

Usage:
  python main.py
  python main.py --age 5 --height 110 --sex M
  python main.py --age 8 --height 50.5 --sex F --unit inches
  python main.py --age 5 --height 110 --sex M --confidence 0.95
"""

import argparse
import sys
from predictor import predict_adult_height, predict_from_inches


def main():
    parser = argparse.ArgumentParser(
        description="Predict adult height from a child's current height (Cole & Wright 2011)"
    )
    parser.add_argument("--age", type=float, help="Child's age in years (2-20)")
    parser.add_argument("--height", type=float, help="Child's current height")
    parser.add_argument("--sex", type=str, help="Sex: M or F")
    parser.add_argument(
        "--unit",
        type=str,
        default="cm",
        choices=["cm", "inches"],
        help="Height unit (default: cm)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.80,
        help="Confidence level for interval (default: 0.80, paper recommends 0.80)",
    )

    args = parser.parse_args()

    # Interactive mode if no arguments
    if args.age is None or args.height is None or args.sex is None:
        print("=" * 55)
        print("  Adult Height Predictor")
        print("  Based on Cole & Wright (2011)")
        print("=" * 55)
        print()

        sex = input("Sex (M/F): ").strip().upper()
        while sex not in ("M", "F"):
            sex = input("Please enter M or F: ").strip().upper()

        age = float(input("Age in years (2-20): ").strip())

        unit = input("Height unit (cm/inches) [cm]: ").strip().lower() or "cm"
        if unit == "inches":
            height = float(input("Current height in inches: ").strip())
        else:
            height = float(input("Current height in cm: ").strip())

        conf_input = input("Confidence level (0.80/0.90/0.95) [0.80]: ").strip()
        confidence = float(conf_input) if conf_input else 0.80
    else:
        sex = args.sex.upper()
        age = args.age
        height = args.height
        unit = args.unit
        confidence = args.confidence

    try:
        if unit == "inches":
            result = predict_from_inches(height, age, sex, confidence)
        else:
            result = predict_adult_height(height, age, sex, confidence)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Display results
    sex_label = "Male" if sex == "M" else "Female"
    conf_pct = int(confidence * 100)

    print()
    print("=" * 55)
    print("  RESULTS")
    print("=" * 55)
    print()
    print(f"  Child:  {sex_label}, age {age} years")
    if unit == "inches":
        print(f"  Height: {height} in ({height * 2.54:.1f} cm)")
    else:
        print(f"  Height: {height} cm ({height / 2.54:.1f} in)")
    print(f"  Percentile: {result['child_percentile']:.0f}th")
    print()
    print(f"  Predicted Adult Height:")
    print(f"    {result['predicted_height_cm']} cm  ({result['predicted_height_ft']})")
    print()
    print(f"  {conf_pct}% Confidence Interval:")
    print(f"    {result['ci_lower_cm']} - {result['ci_upper_cm']} cm")
    print(f"    ({result['ci_lower_ft']}  to  {result['ci_upper_ft']})")
    print()
    print(f"  Standard Error: ±{result['standard_error_cm']} cm")
    print(f"  Correlation (r): {result['correlation']}")
    print()
    print("-" * 55)
    print("  Note: 4 out of 5 predictions fall within ±6 cm")
    print("  of the true adult height (ages 4+).")
    print("  Ref: Cole & Wright, Ann Hum Biol 2011;38:662-8")
    print("-" * 55)


if __name__ == "__main__":
    main()
