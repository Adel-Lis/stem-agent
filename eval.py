import json
import re
import argparse
from openai import OpenAI
from config import OPENAI_API_KEY, MODEL, MODEL_EVAL
from agents.traverser import run_traverser

client = OpenAI(api_key=OPENAI_API_KEY)

TEST_CASES = {
    "deep research": [
        "What are the latest advances in quantum computing?",
        "What are the main approaches researchers use for protein structure prediction?",
        "Summarize the current state of large language model alignment research.",
    ],
    "python analyzer": [
        # bug: h*h*r should be r*r*h
        "Review this function: def cylinder_volume(r, h): return 3.14 * h * h * r",
        # bug: left = mid causes infinite loop, should be left = mid + 1
        (
            "Find any bugs:\n"
            "def binary_search(arr, target):\n"
            "    left, right = 0, len(arr) - 1\n"
            "    while left <= right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target: return mid\n"
            "        elif arr[mid] < target: left = mid\n"
            "        else: right = mid - 1\n"
            "    return -1"
        ),
        # correct — should confirm no bugs
        (
            "Is this implementation correct?\n"
            "def fibonacci(n):\n"
            "    if n <= 0: return []\n"
            "    if n == 1: return [0]\n"
            "    seq = [0, 1]\n"
            "    for i in range(2, n):\n"
            "        seq.append(seq[-2] + seq[-1])\n"
            "    return seq"
        ),
    ],
    "log analyst": [
        (
            "2026-05-01T10:22:11Z ERROR DB connection timeout after 30s\n"
            "2026-05-01T10:22:42Z WARN  Retry attempt 1/3\n"
            "2026-05-01T10:23:13Z ERROR DB connection timeout after 30s\n"
            "2026-05-01T10:23:44Z WARN  Retry attempt 2/3\n"
            "2026-05-01T10:24:15Z FATAL Max retries reached, aborting"
        ),
        (
            "2026-05-01T08:00:01Z INFO  Service started on port 8080\n"
            "2026-05-01T08:05:22Z ERROR NullPointerException in UserService.java:142\n"
            "2026-05-01T08:05:23Z ERROR NullPointerException in UserService.java:142\n"
            "2026-05-01T08:05:24Z FATAL Application crashed — heap dump written"
        ),
        (
            "2026-05-01T12:00:00Z INFO  GET /api/users 200 42ms\n"
            "2026-05-01T12:00:01Z INFO  GET /api/orders 200 38ms\n"
            "2026-05-01T12:00:02Z WARN  GET /api/reports 200 2340ms (threshold: 500ms)\n"
            "2026-05-01T12:00:03Z ERROR GET /api/export 503 Service Unavailable"
        ),
    ],
}

JUDGE_PROMPT = """You are an expert evaluator. Score the AI response below on three dimensions (1-5 each):
- specificity: how domain-specific and detailed is it?
- correctness: is it technically / factually accurate?
- actionability: does it give the user something concrete to act on?

Reply only with JSON: {"specificity": int, "correctness": int, "actionability": int, "reasoning": "one sentence"}"""


def call_baseline(task_input):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": task_input}],
    )
    return resp.choices[0].message.content


def call_judge(task_input, response):
    resp = client.chat.completions.create(
        model=MODEL_EVAL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Task: {task_input}\n\nResponse:\n{response}"},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group()) if m else {"specificity": 0, "correctness": 0, "actionability": 0, "reasoning": "parse error"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=list(TEST_CASES.keys()))
    args = parser.parse_args()

    domain = args.domain
    cases = TEST_CASES[domain]

    baseline_scores = []
    stem_scores = []

    print(f"\nEvaluating: {domain} ({len(cases)} test cases)")

    for i, task_input in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {task_input[:70]}...")

        print("  > baseline running")
        base_score = call_judge(task_input, call_baseline(task_input))
        baseline_scores.append(base_score)

        print("  > stem agent running")
        try:
            stem_resp = run_traverser(domain, task_input)
            stem_score = call_judge(task_input, stem_resp)
        except ValueError as e:
            print(f"  ERROR: {e}")
            print(f'  Run "uv run main.py grow \\"{domain}\\"" first.')
            stem_score = {"specificity": 0, "correctness": 0, "actionability": 0, "reasoning": str(e)}
        stem_scores.append(stem_score)

    metrics = ("specificity", "correctness", "actionability")
    b_avg = {k: round(sum(s[k] for s in baseline_scores) / len(cases), 2) for k in metrics}
    s_avg = {k: round(sum(s[k] for s in stem_scores) / len(cases), 2) for k in metrics}

    print(f"\nResults for '{domain}' — averaged over {len(cases)} inputs (scores 1–5)")
    print(f"{'Metric':<16} {'Baseline':>8} {'Stem Agent':>12} {'Delta':>8}")
    print("-" * 46)
    for m in metrics:
        delta = round(s_avg[m] - b_avg[m], 2)
        print(f"{m:<16} {b_avg[m]:>8} {s_avg[m]:>12} {'+' if delta >= 0 else ''}{delta:>7}")

    b_mean = round(sum(b_avg.values()) / 3, 2)
    s_mean = round(sum(s_avg.values()) / 3, 2)
    d_mean = round(s_mean - b_mean, 2)
    print("-" * 46)
    print(f"{'average':<16} {b_mean:>8} {s_mean:>12} {'+' if d_mean >= 0 else ''}{d_mean:>7}")
