"""Entry point: python -m norvel_writer"""
import sys


def main() -> None:
    from norvel_writer.app import run
    sys.exit(run())


if __name__ == "__main__":
    main()
