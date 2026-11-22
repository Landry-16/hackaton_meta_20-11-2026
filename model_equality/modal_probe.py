"""Decoding-insensitive identity test (Philibert's review, point 5a).

Many short questions with a one-word or one-number answer, packed 12 per prompt and numbered, so that one answer of
the budget buys 12 observations. Per endpoint and question we keep the most frequent answer (the mode). Temperature,
persona, system prompt and watermark change how answers are spread or worded, rarely which answer is most likely;
quantization and LoRA move the model's beliefs and flip the mode on the questions near a decision boundary.

Distance between two endpoints = share of questions (answered by both) where their modes differ.
Same model or not is decided per family with a permutation null (answers of the two endpoints pooled and relabelled),
so the threshold is relative to the noise of that family, never an absolute number.

  python modal_probe.py prompts --out modal_prompts.csv            # write the packed prompts
  python modal_probe.py query --endpoints ref --repeats 10         # ref-* only (unlimited)
  python modal_probe.py score modal_ref_answers.jsonl --practice   # distances, clusters, practice MCC
"""
import argparse
import csv
import itertools
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PER_PROMPT = 12

# (id, type, question). type: num = first number in the answer, word = first content word.
# Medium difficulty for a small model on purpose: questions everyone gets right (or nobody can) carry no signal.
BANK = [
    ("mul1", "num", "What is 37 times 48?"),
    ("mul2", "num", "What is 64 times 29?"),
    ("mul3", "num", "What is 123 times 7?"),
    ("mul4", "num", "What is 56 times 56?"),
    ("add1", "num", "What is 487 plus 365?"),
    ("add2", "num", "What is 1999 plus 2026?"),
    ("sub1", "num", "What is 1000 minus 387?"),
    ("sub2", "num", "What is 803 minus 469?"),
    ("div1", "num", "What is 952 divided by 8?"),
    ("div2", "num", "What is 1001 divided by 7?"),
    ("sq1", "num", "What is the square root of 289?"),
    ("pow1", "num", "What is 2 to the power of 13?"),
    ("pct1", "num", "What is 15 percent of 260?"),
    ("pct2", "num", "What is 35 percent of 80?"),
    ("frac1", "num", "What is one third of 147?"),
    ("prime1", "num", "What is the smallest prime number greater than 90?"),
    ("prime2", "num", "What is the largest prime number below 50?"),
    ("seq1", "num", "What number comes next: 2, 6, 12, 20, 30?"),
    ("seq2", "num", "What number comes next: 1, 1, 2, 3, 5, 8, 13?"),
    ("seq3", "num", "What number comes next: 3, 9, 27, 81?"),
    ("seq4", "num", "What number comes next: 100, 91, 83, 76?"),
    ("cnt1", "num", "How many letters r are in the word strawberry?"),
    ("cnt2", "num", "How many letters s are in the word mississippi?"),
    ("cnt3", "num", "How many letters are in the word elephant?"),
    ("cnt4", "num", "How many vowels are in the word education?"),
    ("legs1", "num", "How many legs do 3 spiders and 2 chickens have in total?"),
    ("days1", "num", "How many days are in February 2024?"),
    ("days2", "num", "How many days are there in 7 weeks?"),
    ("hour1", "num", "How many minutes are in 3.5 hours?"),
    ("sec1", "num", "How many seconds are in 12 minutes?"),
    ("year1", "num", "In which year did the Berlin Wall fall?"),
    ("year2", "num", "In which year did the Titanic sink?"),
    ("year3", "num", "In which year did the French Revolution begin?"),
    ("year4", "num", "In which year was the first iPhone released?"),
    ("year5", "num", "In which year did World War I end?"),
    ("year6", "num", "In which year did Neil Armstrong walk on the Moon?"),
    ("year7", "num", "In which year was the Magna Carta signed?"),
    ("year8", "num", "In which year did Christopher Columbus first reach the Americas?"),
    ("atom1", "num", "What is the atomic number of iron?"),
    ("atom2", "num", "What is the atomic number of gold?"),
    ("atom3", "num", "What is the atomic number of sodium?"),
    ("boil1", "num", "At how many degrees Fahrenheit does water boil at sea level?"),
    ("bone1", "num", "How many bones are in the adult human body?"),
    ("plan1", "num", "How many moons does Mars have?"),
    ("chess1", "num", "How many squares are on a chessboard?"),
    ("keys1", "num", "How many keys does a standard piano have?"),
    ("hex1", "num", "What is the hexadecimal number FF in decimal?"),
    ("bin1", "num", "What is the binary number 1011 in decimal?"),
    ("rom1", "num", "What is the Roman numeral XLIV as a number?"),
    ("rom2", "num", "What is the Roman numeral MCMXC as a number?"),
    ("cap1", "word", "What is the capital of Australia?"),
    ("cap2", "word", "What is the capital of Canada?"),
    ("cap3", "word", "What is the capital of Turkey?"),
    ("cap4", "word", "What is the capital of Switzerland?"),
    ("cap5", "word", "What is the capital of Nigeria?"),
    ("cap6", "word", "What is the capital of Brazil?"),
    ("cap7", "word", "What is the capital of Kazakhstan?"),
    ("cap8", "word", "What is the capital of Morocco?"),
    ("cap9", "word", "What is the capital of New Zealand?"),
    ("cap10", "word", "What is the capital of Vietnam?"),
    ("cap11", "word", "What is the capital of Pakistan?"),
    ("cap12", "word", "What is the capital of Tanzania?"),
    ("sym1", "word", "What is the chemical symbol of silver?"),
    ("sym2", "word", "What is the chemical symbol of tungsten?"),
    ("sym3", "word", "What is the chemical symbol of potassium?"),
    ("sym4", "word", "What is the chemical symbol of lead?"),
    ("week1", "word", "Which day of the week comes three days after Friday?"),
    ("week2", "word", "If today is Wednesday, what day of the week will it be in 10 days?"),
    ("month1", "word", "Which month comes four months after October?"),
    ("month2", "word", "Which is the ninth month of the year?"),
    ("plan2", "word", "Which planet is the largest in the solar system?"),
    ("plan3", "word", "Which planet is closest to the Sun?"),
    ("plan4", "word", "Which planet has the most moons?"),
    ("big1", "word", "Which is larger, 9.11 or 9.9? Answer with the number."),
    ("ocean1", "word", "Which is the largest ocean on Earth?"),
    ("river1", "word", "Which is the longest river in Africa?"),
    ("mount1", "word", "In which country is Mount Kilimanjaro?"),
    ("lang1", "word", "What is the official language of Brazil?"),
    ("curr1", "word", "What is the currency of Japan?"),
    ("curr2", "word", "What is the currency of Switzerland?"),
    ("auth1", "word", "Who wrote the novel 1984? Give the surname."),
    ("auth2", "word", "Who painted the Mona Lisa? Give the surname."),
    ("auth3", "word", "Who wrote Pride and Prejudice? Give the surname."),
    ("auth4", "word", "Who composed the Fifth Symphony that starts with four famous notes? Give the surname."),
    ("sci1", "word", "Who proposed the theory of general relativity? Give the surname."),
    ("sci2", "word", "Who discovered penicillin? Give the surname."),
    ("opp1", "word", "What is the opposite of the word ancient?"),
    ("opp2", "word", "What is the opposite of the word generous?"),
    ("syn1", "word", "Give one synonym of the word rapid."),
    ("syn2", "word", "Give one synonym of the word huge."),
    ("anim1", "word", "What is the fastest land animal?"),
    ("anim2", "word", "What is the largest mammal?"),
    ("anim3", "word", "What is a baby kangaroo called?"),
    ("col1", "word", "What color do you get by mixing blue and yellow?"),
    ("col2", "word", "What color do you get by mixing red and white?"),
    ("gas1", "word", "Which gas do plants absorb from the air for photosynthesis? Give its chemical formula."),
]
TYPE = {q: t for q, t, _ in BANK}
HEADER = ("Answer each question with one word or one number only. Write one line per question in the form "
          "'N. answer', with no explanation.\n\n")


def packed(seed=0):
    """Fixed packing: shuffled once with a fixed seed so related questions are spread over prompts."""
    items = BANK[:]
    random.Random(seed).shuffle(items)
    prompts = []
    for k in range(0, len(items), PER_PROMPT):
        chunk = items[k:k + PER_PROMPT]
        text = HEADER + "\n".join(f"{i + 1}. {q}" for i, (_, _, q) in enumerate(chunk))
        prompts.append({"id": f"modal-{k // PER_PROMPT:02d}", "qids": [q for q, _, _ in chunk], "prompt": text})
    return prompts


NUM = re.compile(r"-?\d+(?:[.,]\d+)?")
STOP = {"the", "a", "an", "is", "it", "its", "answer", "of", "approximately", "about"}


def norm(answer, kind):
    a = answer.strip().lower().replace("**", "")
    if kind == "num":
        m = NUM.search(a.replace(",", "") if re.search(r"\d,\d{3}", a) else a)
        if not m:
            return None
        v = m.group(0).replace(",", ".")
        return v[:-2] if v.endswith(".0") else v
    words = [w for w in re.findall(r"[a-z0-9.]+", a) if w not in STOP]
    return words[0].strip(".") if words else None


LINE = re.compile(r"^\s*(?:\*\*)?(?:(?:n\.\s*)?(?:answer|question)\s*)?(\d{1,2})\s*[.):\-]\s*(?:\*\*)?\s*(.+)$", re.I)


def parse(response, qids):
    """{qid: normalized answer} for the numbered lines we can read; missing lines are left out."""
    out = {}
    for line in response.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        n = int(m.group(1)) - 1
        if 0 <= n < len(qids) and qids[n] not in out:
            ans = re.sub(r"^\s*n\.?\s*answer\s*[:\-]?\s*", "", m.group(2), flags=re.I)  # echoed the template
            ans = re.sub(r"^.*?\?\s*", "", ans) if "?" in ans[:-1] else ans   # model repeated the question
            if ":" in ans and not re.search(r"\d:\d", ans):                   # "Capital of Nigeria: Abuja"
                ans = ans.rsplit(":", 1)[1]
            ans = re.sub(r"\(.*?\)", "", ans).strip(" '\"")                    # "384 (13 x 28)"
            v = norm(ans, TYPE[qids[n]])
            if v:
                out[qids[n]] = v
    return out


def load(path):
    """{endpoint: {qid: [answers]}} from a jsonl written by `query`."""
    qids = {p["id"]: p["qids"] for p in packed()}
    data = defaultdict(lambda: defaultdict(list))
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"] not in qids:
            continue
        for q, v in parse(r["response"], qids[r["id"]]).items():
            data[r["endpoint"]][q].append(v)
    return data


def mode_set(answers):
    c = Counter(answers)
    top = max(c.values())
    return {a for a, n in c.items() if n == top}


def distance(a, b, min_n=2):
    """Share of shared questions where the modes differ (ties: differ only if the tied sets do not overlap)."""
    qs = [q for q in a if q in b and len(a[q]) >= min_n and len(b[q]) >= min_n]
    if not qs:
        return 1.0, 0
    diff = sum(1 for q in qs if not (mode_set(a[q]) & mode_set(b[q])))
    return diff / len(qs), len(qs)


def tv_distance(a, b, min_n=2):
    """Mean total-variation distance between the answer distributions (more power, more sensitive to temperature)."""
    qs = [q for q in a if q in b and len(a[q]) >= min_n and len(b[q]) >= min_n]
    if not qs:
        return 1.0, 0
    tot = 0.0
    for q in qs:
        ca, cb = Counter(a[q]), Counter(b[q])
        tot += 0.5 * sum(abs(ca[k] / len(a[q]) - cb[k] / len(b[q])) for k in set(ca) | set(cb))
    return tot / len(qs), len(qs)


STAT = {"mode": distance, "tv": tv_distance}


def perm_pvalue(a, b, n_perm=200, seed=0, stat="mode"):
    """P(distance >= observed) when the answers of a and b are pooled per question and randomly split again."""
    distance = STAT[stat]
    obs, n = distance(a, b)
    if n == 0:
        return 1.0, obs
    rng = random.Random(seed)
    qs = [q for q in a if q in b]
    hits = 0
    for _ in range(n_perm):
        pa, pb = {}, {}
        for q in qs:
            pool = a[q] + b[q]
            rng.shuffle(pool)
            pa[q], pb[q] = pool[:len(a[q])], pool[len(a[q]):]
        hits += distance(pa, pb)[0] >= obs
    return (hits + 1) / (n_perm + 1), obs


def cluster(eps, same):
    """Groups where every pair is called same (greedy complete linkage on the order of strongest evidence)."""
    groups = [{e} for e in eps]
    pairs = sorted(same, key=lambda p: same[p], reverse=True)
    for (a, b) in pairs:
        ga = next(g for g in groups if a in g)
        gb = next(g for g in groups if b in g)
        if ga is gb:
            continue
        if all(frozenset((x, y)) in same for x in ga for y in gb):
            groups.remove(gb)
            ga |= gb
    return groups


def decide(data, eps, alpha, stat="mode"):
    """Pair is same when the permutation test does not reject equality at level alpha."""
    rows, same = [], {}
    for a, b in itertools.combinations(eps, 2):
        p, d = perm_pvalue(data[a], data[b], stat=stat)
        rows.append((a, b, d, p))
        if p > alpha:
            same[frozenset((a, b))] = p
    return rows, cluster(eps, same)


def practice_mcc(groups, truth_path=ROOT / "practice_answers.csv"):
    truth = {r["endpoint"]: r["model"] for r in csv.DictReader(open(truth_path))}
    label = {e: i for i, g in enumerate(groups) for e in g}
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(sorted(truth), 2):
        t, g = truth[a] == truth[b], label[a] == label[b]
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    d = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (max(0.0, (tp * tn - fp * fn) / d) if d else 0.0), (tp, fp, fn, tn)


def cmd_prompts(args):
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prompt"])
        for p in packed():
            w.writerow([p["id"], p["prompt"]])
    print(f"{len(packed())} prompts of {PER_PROMPT} questions -> {args.out}")


def cmd_query(args):
    if args.endpoints != "ref":
        sys.exit("This command only queries ref-*. Scored endpoints go through modal_ep_batch.py.")
    import query_reference as qr
    prompts = [(p["id"], p["prompt"]) for p in packed()]
    if args.prompts:
        prompts = prompts[:args.prompts]
    qr.query(prompts, output=Path(args.out), repeats=args.repeats, max_tokens=args.max_tokens)


def cmd_score(args):
    data = load(args.answers)
    eps = sorted(e for e in data if (args.only is None or e in args.only.split(",")))
    if args.subsample:
        rng = random.Random(args.seed)
        for e in eps:
            for q in data[e]:
                if len(data[e][q]) > args.subsample:
                    data[e][q] = rng.sample(data[e][q], args.subsample)
    if args.stable:
        # keep questions where, pooled over the family, the top answer holds at least this share (no labels used)
        keep = set()
        for q in {q for e in eps for q in data[e]}:
            pool = [v for e in eps for v in data[e].get(q, [])]
            if pool and Counter(pool).most_common(1)[0][1] / len(pool) >= args.stable:
                keep.add(q)
        for e in eps:
            data[e] = defaultdict(list, {q: v for q, v in data[e].items() if q in keep})
        print(f"stable questions (top share >= {args.stable}): {len(keep)}")
    cover = {e: (len(data[e]), sum(map(len, data[e].values()))) for e in eps}
    print("coverage (questions, answers):", " ".join(f"{e}={c[0]}/{c[1]}" for e, c in cover.items()))
    rows, groups = decide(data, eps, args.alpha, args.stat)
    if args.verbose:
        for a, b, d, p in sorted(rows, key=lambda r: r[2]):
            print(f"  {a} {b}  dist {d:.3f}  p {p:.3f}")
    print("groups:", [sorted(g) for g in groups])
    if args.practice:
        mcc, cm = practice_mcc(groups)
        print(f"practice MCC {mcc:.3f}  (tp fp fn tn) {cm}")
    return groups


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts"); p.add_argument("--out", default="modal_prompts.csv")
    q = sub.add_parser("query"); q.add_argument("--endpoints", default="ref")
    q.add_argument("--repeats", type=int, default=10); q.add_argument("--prompts", type=int)
    q.add_argument("--out", default="modal_ref_answers.jsonl"); q.add_argument("--max-tokens", type=int, default=128)
    s = sub.add_parser("score"); s.add_argument("answers"); s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--practice", action="store_true"); s.add_argument("--verbose", action="store_true")
    s.add_argument("--only"); s.add_argument("--subsample", type=int); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--stat", choices=sorted(STAT), default="mode")
    s.add_argument("--stable", type=float, help="keep questions whose pooled top answer share is at least this")
    args = ap.parse_args()
    {"prompts": cmd_prompts, "query": cmd_query, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
