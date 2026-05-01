# Evaluation Report | Stem Agent vs. Vanilla LLM Baseline

> Report by Adel Lis

---

## What is eval.py and why it exists

The core claim of the stem agent is that a self-grown, domain-specialized pipeline produces better outputs than a raw LLM call on the same input. To make that claim measurable, I wrote `eval.py`, a script that runs both systems on the same set of inputs and scores each output with an LLM judge.

To run the evaluation for a specific domain:

```bash
uv run eval.py "python analyzer"
uv run eval.py "deep research"
uv run eval.py "log analyst"
```

> Make sure the Stem Agents EXIST in `output/` folder and each HAVE TEST CASES in the `TEST_CASES` dictionary (in eval.py)

---

## Methodology

**Baseline** : the same model the stem agent uses internally (`o4-mini`), called with no system prompt and no tools. Just the raw user query.

**Stem Agent** : the fully grown domain-specialized pipeline loaded from checkpoint. The traverser walks the graph in topological order, each node processes the input, and the final strategy node returns the answer.

**Judge** : `gpt-4.1` scores each response independently on three dimensions (1 – 5):

- **Specificity**: how domain-specific and detailed is the answer?
- **Correctness**: is it technically or factually accurate?
- **Actionability**: does it give the user something concrete?

Each domain has 3 test inputs, the scores reported are averages across those 3 inputs. The judge sees one response at a time and scores it without knowing which system produced it.

Results are presented as a table with three columns: **Baseline** (raw LLM score), **Stem Agent** (specialized pipeline score), and **Delta** (Stem Agent - Baseline). _A positive delta means the stem agent scored higher than the baseline on that dimension; a negative delta means the baseline did better. The larger the absolute delta, the more the two systems diverged on that metric._

---

## Experiment Setup

For this experiment, I have three stem agents (you can check an example of each of them in `output-local/` folder), for each agent I hard-coded 3 test cases in the eval.py file. The three stem agents are:

**python analyzer**, tested with three code review tasks: one function with a formula bug (`h*h*r` instead of `r*r*h`), one with an infinite loop bug (`left = mid` instead of `left = mid + 1`), and one correct fibonacci implementation that should be confirmed as correct.

**deep research**, tested with three open-ended research questions on quantum computing, protein structure prediction, and LLM alignment.

**log analyst**, tested with three raw log dumps covering different failure scenarios: database timeouts with retries, a fatal crash from a NullPointerException, and a slow endpoint with an occasional 503.

---

## Results

### python analyzer

| Metric        | Baseline | Stem Agent | Delta   |
| ------------- | -------- | ---------- | ------- |
| specificity   | 5.0      | 5.0        | 0.0     |
| correctness   | 5.0      | 5.0        | 0.0     |
| actionability | 5.0      | 5.0        | 0.0     |
| **average**   | **5.0**  | **5.0**    | **0.0** |

Both systems scored perfectly. Apparently reviewing code is very easy to do for a baseline LLM, probably due to the high amount of information online. The Stem Agent scored perfectly, which is really great. A more specific experiment with this agent would be testing how both perform when analyzing entire applications.

---

### log analyst

| Metric        | Baseline | Stem Agent | Delta     |
| ------------- | -------- | ---------- | --------- |
| specificity   | 4.67     | 5.0        | +0.33     |
| correctness   | 5.0      | 5.0        | 0.0       |
| actionability | 5.0      | 5.0        | 0.0       |
| **average**   | **4.89** | **5.0**    | **+0.11** |

Small but consistent improvement are observed, the Stem Agent did better. The `parse_logs` tool the agent grew during specialization extracts structured records from raw log lines before passing them to the strategy node. This preprocessing step is what makes the Stem Agent more specialized. The baseline has to interpret raw text directly, while the stem agent works on structured output. The gain is modest here because the baseline is already strong on log analysis.

---

### deep research

| Metric        | Baseline | Stem Agent | Delta     |
| ------------- | -------- | ---------- | --------- |
| specificity   | 5.0      | 3.67       | -1.33     |
| correctness   | 5.0      | 3.67       | -1.33     |
| actionability | 3.33     | 3.33       | 0.0       |
| **average**   | **4.44** | **3.56**   | **-0.88** |

The stem agent performed worse here. The baseline (`o4-mini`) answers research questions directly from its parametric knowledge, which is broad and reliable for topics like quantum computing and LLM alignment. The stem agent relies on live web search results via its `web_search` tool. Upon deeper investigation, Web search introduces noise and the results can end up being inconsistent, off-topic, or of lower quality than the model's own knowledge on well-documented topics. The pipeline also passes the search output through several strategy nodes before producing a final answer, and each transformation is an opportunity to lose precision or introduce vagueness.

This does not mean the pipeline design is wrong, on other tests during development on topics that are more niche or recent (where the model has weaker parametric knowledge), the web-augmented search outperform the baseline. But for well-covered research domains, the added complexity does not bring any good. To be honest, every time I would run this stem agent I would have good and bad answers about 50% of the times; in the end the evaluation tests are right, here it depends on the topic to research, more than on the method to answer.

---

## Summary

| Domain          | Baseline avg | Stem Agent avg | Delta |
| --------------- | ------------ | -------------- | ----- |
| python analyzer | 5.0          | 5.0            | 0.0   |
| log analyst     | 4.89         | 5.0            | +0.11 |
| deep research   | 4.44         | 3.56           | -0.88 |

The results are mixed, the stem agent matches the baseline on structured tasks (code review), improves slightly when preprocessing is useful (log analysis), and underperforms when the baseline's parametric knowledge is stronger than what live web search can provide (deep research). This evaluation makes me realize a real issue that would need to be addressed: the specialization check (`is_fully_specialized`) only verifies graph structure, not output quality. A stem agent can be structurally complete and still perform worse than the baseline if the tools it grows introduce noise rather than signal.

I would also like to note that these are the results of this specific evaluation run. A lot of times the Stem Agent crushes the Baseline model, even the deep research, but in most cases the results are like the ones in this report. What I want to say is that you might want to run the evaluation multiple times and see yourself what are the results. I ended up evaluating my agents, then I deleted and re-created new agents (the `grow` in main) which performed better than the first ones. Evaluation can be tidious, so doing lots of experiments is useful to have a bigger progect of how the Stem Agents are performing.
