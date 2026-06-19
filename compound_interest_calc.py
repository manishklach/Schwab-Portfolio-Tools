#!/usr/bin/env python3
"""Generic compound-interest calculator with reinvestment."""

from __future__ import annotations

import argparse


def build_schedule(
    principal: float,
    annual_yield: float,
    years: float,
    monthly_contribution: float,
) -> list[dict[str, float]]:
    months = max(int(round(years * 12)), 0)
    monthly_rate = annual_yield / 12.0
    balance = principal
    total_contributions = principal
    total_interest = 0.0
    schedule: list[dict[str, float]] = []

    for month in range(1, months + 1):
        interest = balance * monthly_rate
        balance += interest
        total_interest += interest

        if monthly_contribution:
            balance += monthly_contribution
            total_contributions += monthly_contribution

        schedule.append(
            {
                "month": month,
                "interest": interest,
                "contribution": monthly_contribution,
                "ending_balance": balance,
                "total_interest": total_interest,
                "total_contributions": total_contributions,
            }
        )

    return schedule


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generic compound-interest calculator with reinvestment."
    )
    parser.add_argument("--principal", type=float, required=True, help="Starting principal amount.")
    parser.add_argument(
        "--annual-yield",
        type=float,
        required=True,
        help="Annual yield as a decimal, e.g. 0.05 for 5%%.",
    )
    parser.add_argument("--years", type=float, required=True, help="Investment horizon in years.")
    parser.add_argument(
        "--monthly-contribution",
        type=float,
        default=0.0,
        help="Additional monthly contribution added after interest each month.",
    )
    parser.add_argument(
        "--show-monthly",
        action="store_true",
        help="Print the monthly compounding schedule.",
    )
    args = parser.parse_args()

    schedule = build_schedule(
        principal=args.principal,
        annual_yield=args.annual_yield,
        years=args.years,
        monthly_contribution=args.monthly_contribution,
    )

    months = int(round(args.years * 12))
    ending_balance = schedule[-1]["ending_balance"] if schedule else args.principal
    total_interest = schedule[-1]["total_interest"] if schedule else 0.0
    total_contributions = schedule[-1]["total_contributions"] if schedule else args.principal

    print("\nCompound Interest Estimate")
    print("=" * 72)
    print(f"Starting Principal:      ${args.principal:,.2f}")
    print(f"Annual Yield:            {args.annual_yield * 100:.3f}%")
    print(f"Horizon:                 {args.years:.2f} years ({months} months)")
    print(f"Monthly Contribution:    ${args.monthly_contribution:,.2f}")
    print(f"Total Contributions:     ${total_contributions:,.2f}")
    print(f"Total Interest Earned:   ${total_interest:,.2f}")
    print(f"Ending Balance:          ${ending_balance:,.2f}")

    if args.show_monthly and schedule:
        print("\nMonthly Schedule")
        print(f"{'Month':>5} {'Interest':>14} {'Contribution':>14} {'Ending Balance':>18}")
        for row in schedule:
            print(
                f"{int(row['month']):>5} "
                f"{row['interest']:>14,.2f} "
                f"{row['contribution']:>14,.2f} "
                f"{row['ending_balance']:>18,.2f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
