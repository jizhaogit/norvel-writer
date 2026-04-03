"""Entry point: python -m norvel_writer"""
import os
import sys

# Disable ChromaDB's PostHog telemetry before chromadb is imported anywhere.
# Their current PostHog integration has a broken call signature that spams the
# log with "capture() takes 1 positional argument but 3 were given".
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")


def main() -> None:
    from norvel_writer.app import run
    sys.exit(run())


if __name__ == "__main__":
    main()
