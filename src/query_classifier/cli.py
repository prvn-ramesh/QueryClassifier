import sys
import json
import argparse
from pathlib import Path

from query_classifier.classifier import QueryClassifier


def main():
    parser = argparse.ArgumentParser(
        prog="query-classifier",
        description="Fast ONNX query difficulty classifier for pre-routing in LLM cascades.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        help="Text query to classify (e.g. 'What is the capital of France?')",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to a text file containing queries (one per line)",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        help="Path to local directory containing ONNX model files",
    )
    parser.add_argument(
        "--hf-repo",
        "-r",
        type=str,
        help="Hugging Face hub repo ID to download ONNX files from",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output raw JSON format",
    )

    args = parser.parse_args()

    if not args.query and not args.file:
        parser.print_help()
        sys.exit(1)

    try:
        classifier = QueryClassifier(model_path=args.model_path, hf_repo=args.hf_repo)
    except Exception as e:
        print(f"Error initializing classifier: {e}", file=sys.stderr)
        sys.exit(1)

    if args.query:
        result = classifier.predict(args.query)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Query:       \"{args.query}\"")
            print(f"Difficulty:  {result.label.upper()}")
            print(f"Confidence:  {result.confidence:.4f}")
            print(f"Latency:     {result.latency_ms:.2f} ms")

    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File '{args.file}' not found.", file=sys.stderr)
            sys.exit(1)

        with open(file_path, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]

        results = classifier.predict_batch(queries)
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            for q, r in zip(queries, results):
                print(f"[{r.label.upper():6s}] ({r.confidence:.2f} | {r.latency_ms:.1f}ms) -> {q}")


if __name__ == "__main__":
    main()
