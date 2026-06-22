const data = JSON.parse(document.getElementById("panorama-data").textContent);
const trackFilter = document.getElementById("track-filter");
const searchInput = document.getElementById("search");

const trackById = new Map(data.tracks.map((track) => [track.id, track]));
const referenceById = new Map(data.references.map((reference) => [reference.id, reference]));

function init() {
  renderMetrics();
  renderTrackOptions();
  render();
  searchInput.addEventListener("input", render);
  trackFilter.addEventListener("change", render);
}

function renderMetrics() {
  const tags = new Set(data.nodes.flatMap((node) => node.tags));
  const metrics = [
    ["tracks", data.tracks.length],
    ["nodes", data.nodes.length],
    ["edges", data.edges.length],
    ["sources", data.references.length],
    ["tags", tags.size],
    ["version", data.meta.version],
  ];
  document.getElementById("metrics").innerHTML = metrics
    .map(([label, value]) => `<div class="metric"><dt>${label}</dt><dd>${value}</dd></div>`)
    .join("");
}

function renderTrackOptions() {
  for (const track of data.tracks) {
    const option = document.createElement("option");
    option.value = track.id;
    option.textContent = track.title;
    trackFilter.appendChild(option);
  }
}

function render() {
  const query = searchInput.value.trim().toLowerCase();
  const selectedTrack = trackFilter.value;
  const nodes = data.nodes.filter((node) => {
    const inTrack = selectedTrack === "all" || node.track === selectedTrack;
    const haystack = [
      node.title,
      node.summary,
      node.why,
      node.phase,
      node.status,
      ...node.tags,
      ...node.references.map((id) => referenceById.get(id)?.title ?? ""),
    ]
      .join(" ")
      .toLowerCase();
    return inTrack && (!query || haystack.includes(query));
  });

  renderTracks(nodes);
  renderGraph(nodes);
  renderReferences(nodes);
}

function renderTracks(nodes) {
  const nodeIds = new Set(nodes.map((node) => node.id));
  const markup = data.tracks
    .map((track) => {
      const trackNodes = data.nodes.filter((node) => node.track === track.id && nodeIds.has(node.id));
      if (trackNodes.length === 0) {
        return "";
      }
      return `
        <article class="track">
          <div class="track-header">
            <div>
              <p class="eyebrow">${track.id}</p>
              <h2>${escapeHtml(track.title)}</h2>
            </div>
            <p>${escapeHtml(track.summary)}</p>
          </div>
          <div class="cards">
            ${trackNodes.map(renderCard).join("")}
          </div>
        </article>
      `;
    })
    .join("");
  document.getElementById("tracks").innerHTML = markup || `<p>No matching nodes.</p>`;
}

function renderCard(node) {
  const referenceLinks = node.references
    .map((id) => referenceById.get(id))
    .filter(Boolean)
    .map((reference) => `<a class="pill" href="${reference.url}" target="_blank" rel="noreferrer">${reference.year}</a>`)
    .join("");
  return `
    <article class="card" id="${node.id}">
      <h3>${escapeHtml(node.title)}</h3>
      <p>${escapeHtml(node.summary)}</p>
      <p><strong>Why it matters:</strong> ${escapeHtml(node.why)}</p>
      <div class="card-meta">
        <span class="pill status-${node.status}">${escapeHtml(node.status)}</span>
        <span class="pill">${escapeHtml(node.phase)}</span>
        ${node.tags.map((tag) => `<span class="pill">${escapeHtml(tag)}</span>`).join("")}
        ${referenceLinks}
      </div>
    </article>
  `;
}

function renderGraph(nodes) {
  const svg = document.getElementById("graph");
  const visible = new Set(nodes.map((node) => node.id));
  const visibleNodes = data.nodes.filter((node) => visible.has(node.id));
  const visibleEdges = data.edges.filter((edge) => visible.has(edge.source) && visible.has(edge.target));
  const width = 1120;
  const height = 620;
  const trackOrder = new Map(data.tracks.map((track, index) => [track.id, index]));
  const grouped = new Map();

  for (const node of visibleNodes) {
    if (!grouped.has(node.track)) {
      grouped.set(node.track, []);
    }
    grouped.get(node.track).push(node);
  }

  const positions = new Map();
  for (const [trackId, trackNodes] of grouped) {
    const row = trackOrder.get(trackId) ?? 0;
    const y = 68 + row * 88;
    trackNodes.forEach((node, index) => {
      const x = 130 + index * Math.max(142, Math.floor(840 / Math.max(trackNodes.length, 1)));
      positions.set(node.id, { x: Math.min(x, width - 120), y });
    });
  }

  const edgeMarkup = visibleEdges
    .map((edge) => {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) return "";
      return `<line class="graph-edge" x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}"><title>${escapeHtml(edge.relation)}: ${escapeHtml(edge.summary)}</title></line>`;
    })
    .join("");

  const nodeMarkup = visibleNodes
    .map((node) => {
      const point = positions.get(node.id);
      if (!point) return "";
      const color = trackColor(node.track);
      return `
        <g class="graph-node" transform="translate(${point.x} ${point.y})" tabindex="0" role="link" aria-label="${escapeHtml(node.title)}">
          <circle r="19" fill="${color}"></circle>
          <text x="0" y="40" text-anchor="middle">${escapeHtml(shortLabel(node.title))}</text>
          <title>${escapeHtml(node.title)}: ${escapeHtml(node.summary)}</title>
        </g>
      `;
    })
    .join("");

  const trackLabels = data.tracks
    .map((track, index) => {
      const y = 72 + index * 88;
      return `<text x="24" y="${y}" fill="#64748b" font-size="13" font-weight="800">${escapeHtml(track.title)}</text>`;
    })
    .join("");

  svg.innerHTML = `${trackLabels}${edgeMarkup}${nodeMarkup}`;
  svg.querySelectorAll(".graph-node").forEach((group) => {
    group.addEventListener("click", () => {
      const title = group.querySelector("title").textContent.split(":")[0];
      const node = data.nodes.find((item) => item.title === title);
      if (node) {
        document.getElementById(node.id)?.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
  });
}

function renderReferences(nodes) {
  const referenceIds = new Set(nodes.flatMap((node) => node.references));
  const references = data.references.filter((reference) => referenceIds.has(reference.id));
  document.getElementById("references").innerHTML = references
    .map(
      (reference) => `
      <article class="reference">
        <a href="${reference.url}" target="_blank" rel="noreferrer">${escapeHtml(reference.title)}</a>
        <p>${reference.year} · ${escapeHtml(reference.kind)}</p>
      </article>
    `
    )
    .join("");
}

function trackColor(trackId) {
  const palette = {
    alignment: "#38bdf8",
    reasoning_rl: "#6366f1",
    fusion: "#14b8a6",
    agentic_rl: "#f59e0b",
    systems: "#0f766e",
    evaluation_safety: "#ef4444",
  };
  return palette[trackId] ?? "#64748b";
}

function shortLabel(value) {
  return value.length > 18 ? `${value.slice(0, 16)}...` : value;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

init();

