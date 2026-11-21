"""Build a self-contained, script-free HTML comparison with SVG projections."""

from html import escape
import json


COLORS = ("#158c84", "#c66d36", "#6f62ad", "#cc5273", "#647b2c", "#2679aa", "#a86b18", "#8c4b70")


def scatter(points, model, title, bounds):
    lower, upper = bounds

    def project(point):
        x, y = [30 + 440 * (point[a] - lower[a]) / max(upper[a] - lower[a], 1e-9) for a in (0, 1)]
        return x, 500 - y

    shapes = []
    for point, label in zip(points, model["labels"]):
        x, y = project(point)
        shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="{COLORS[label % len(COLORS)]}" opacity=".5"/>')
    for point in model["centroids"]:
        x, y = project(point)
        shapes.append(f'<path d="M{x-5:.2f},{y-5:.2f}l10,10m-10,0l10,-10" stroke="#172b32" stroke-width="3"/>')
    return f'<svg viewBox="0 0 500 500" role="img" aria-label="{escape(title)} projected clusters"><rect x="1" y="1" width="498" height="498" rx="15" fill="#f4f7f7"/>{"".join(shapes)}</svg>'


def render_report(summary, plots):
    config = summary["config"]
    rows, panels = [], []
    all_points = plots["points"] + [point for name in summary["models"] for point in plots[name]["centroids"]]
    bounds = ([min(p[a] for p in all_points) for a in (0, 1)],
              [max(p[a] for p in all_points) for a in (0, 1)])
    for name, model in summary["models"].items():
        rows.append(f'<tr><th scope="row">{escape(name)}</th><td>{model["train_mse"]:.4f}</td><td>{model["heldout_mse"]:.4f}</td><td>{model["training_seconds"]:.4f}s</td><td>{model["evaluation_seconds"]:.4f}s</td></tr>')
        panels.append(f'<section><h2>{escape(name)}</h2>{scatter(plots["points"], plots[name], name, bounds)}</section>')
    recipe = escape(json.dumps(summary["emeans_recipe"], indent=2))
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>e-means clustering benchmark</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#172b32;background:#eaf0ef;margin:0}}main{{max-width:1100px;margin:auto;padding:35px 24px}}h1{{font-size:clamp(28px,5vw,48px);line-height:1.1}}.eyebrow{{font-weight:700;letter-spacing:.12em;color:#158c84}}.card{{background:white;padding:24px;border-radius:18px;margin:20px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:24px}}svg{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #dbe4e2}}.scroll{{overflow:auto}}pre{{overflow:auto;background:#f4f7f7;padding:16px}}a{{color:#08746e}}footer{{font-size:14px}}</style>
<main><p class="eyebrow">NATIVE FAISS · CPU</p><h1>More updates per pass.<br>Measure the final result.</h1>
<p>Compare e-means with Lloyd clustering on {config["samples"]:,} synthetic vectors, {config["dimension"]} dimensions and {config["clusters"]} clusters. Both start from identical centroids and see the full training dataset.</p>
<div class="card scroll"><table><caption>Final exact nearest-centroid squared L2 loss per vector</caption><thead><tr><th>Method</th><th>Training loss</th><th>Held-out loss</th><th>Training time</th><th>Evaluation time</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<p>e-means: {config["epochs"]:g} passes. Lloyd: {config["lloyd_epochs"]} passes. Times are one local measurement with {config["threads"]} Faiss thread(s), without warm-up or repeated trials. Evaluation time includes final assignments on training and held-out data.</p></div>
<div class="card"><div class="grid">{"".join(panels)}</div><p>First two coordinates of up to 600 training vectors, colored by exact final assignments. Crosses mark final centroids. Loss uses every coordinate; the projection can hide separation. Cluster color identities are arbitrary.</p></div>
<div class="card"><h2>What the final assignment changes</h2><p>e-means returns labels from the last moving training pass. Here {summary["moving_labels_differ_from_final"]:,} differ from assignments to the final centroids. This report recomputes labels for both algorithms with the same exact L2 evaluator.</p>
<details><summary>Native e-means recipe and revival count ({summary["emeans_revivals"]})</summary><pre>{recipe}</pre></details></div>
<footer><p>Faiss {escape(summary["faiss_version"])} · synthetic data · no GPU speedup claim. This small CPU comparison does not reproduce published large-scale benchmarks.</p><p><a href="results.json">Download measured results</a> · <a href="https://github.com/facebookresearch/faiss/tree/main/contrib/e_means">Upstream implementation</a></p></footer></main></html>'''
