from src.policies import stick_on_20_policy, STICK, HIT


def main():
    test_states = [
        (16, 8, 0),
        (19, 10, 0),
        (20, 7, 0),
        (21, 5, 1),
    ]

    for state in test_states:
        action = stick_on_20_policy(state)

        print(
            f"État : {state} -> "
            f"Action : {action}"
        )


if __name__ == "__main__":
    main()