import {
  Callout,
  Divider,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Text,
  computeDAGLayout,
  useHostTheme,
  useState,
} from "cursor/canvas";

type FlowNode = { id: string; label: string; sub: string; hub?: boolean };
type FlowEdge = { from: string; to: string; dashed?: boolean };

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "m1", label: "Module 1" },
  { id: "m2w12", label: "Module 2 · Week 1–2" },
  { id: "m2w3", label: "Module 2 · Week 3" },
  { id: "handoff", label: "Handoff" },
] as const;

type TabId = (typeof TABS)[number]["id"];

const GRAPHS: Record<
  TabId,
  { title: string; blurb: string; nodes: FlowNode[]; edges: FlowEdge[]; direction: "horizontal" | "vertical" }
> = {
  overview: {
    title: "End-to-end (Modules 1 → 2 → 3)",
    blurb: "Data Acquisition and Preprocessing are complete. Reconstruction consumes filtered S21 today; Hamming / time-domain arrive via handoff export.",
    direction: "horizontal",
    nodes: [
      { id: "file", label: ".mat / .sNp", sub: "datasets/" },
      { id: "m1", label: "Module 1", sub: "load + validate", hub: true },
      { id: "ds", label: "MicrowaveDataset", sub: "canonical model", hub: true },
      { id: "route", label: "Module 2 router", sub: ".s2p vs general" },
      { id: "s21", label: "S21 UG track", sub: "Week 1–3" },
      { id: "gen", label: "General track", sub: "mat multi-trace" },
      { id: "out", label: "PreprocessingResult", sub: "+ Week 3 meta", hub: true },
      { id: "exp", label: "Handoff export", sub: ".mat / csv / json" },
      { id: "m3", label: "Module 3", sub: "DAS · DMAS · D4" },
    ],
    edges: [
      { from: "file", to: "m1" },
      { from: "m1", to: "ds" },
      { from: "ds", to: "route" },
      { from: "route", to: "s21" },
      { from: "route", to: "gen" },
      { from: "s21", to: "out" },
      { from: "gen", to: "out" },
      { from: "out", to: "exp" },
      { from: "out", to: "m3" },
      { from: "exp", to: "m3", dashed: true },
      { from: "ds", to: "m3", dashed: true },
    ],
  },
  m1: {
    title: "Module 1 — Data Acquisition",
    blurb: "Unified loader returns MicrowaveDataset. BMID cubes open a scan picker before forming one (1001 × 72) scan.",
    direction: "vertical",
    nodes: [
      { id: "pick", label: "Load Dataset", sub: "UploadPage GUI" },
      { id: "disp", label: "load_dataset()", sub: "loader.py", hub: true },
      { id: "mat", label: "MATLAB path", sub: "legacy + HDF5 v7.3" },
      { id: "bmid", label: "BMID extract", sub: "scan picker + md_list" },
      { id: "ts", label: "Touchstone path", sub: "scikit-rf + S21 helpers" },
      { id: "val", label: "Validate", sub: "freqs · S-params · finite" },
      { id: "sum", label: "DatasetSummary", sub: "GUI table" },
      { id: "out", label: "MicrowaveDataset", sub: "dataset_loaded signal", hub: true },
    ],
    edges: [
      { from: "pick", to: "disp" },
      { from: "disp", to: "mat" },
      { from: "disp", to: "bmid" },
      { from: "disp", to: "ts" },
      { from: "mat", to: "val" },
      { from: "bmid", to: "val" },
      { from: "ts", to: "val" },
      { from: "val", to: "sum" },
      { from: "sum", to: "out" },
    ],
  },
  m2w12: {
    title: "Module 2 — Week 1–2 (.s2p S21)",
    blurb: "Complex linear S21 only. Averaging / filtering never run on dB values. NRMSE flags over-smoothing.",
    direction: "vertical",
    nodes: [
      { id: "in", label: "Valid .s2p", sub: "S21 extract only" },
      { id: "hdr", label: "Header + Hz", sub: "DB / MA / RI → complex" },
      { id: "chk", label: "Validate + sort", sub: "Δf · no NaN/dupes" },
      { id: "uni", label: "Uniform grid", sub: "interp R & I if needed" },
      { id: "plot", label: "Before plots", sub: "|S21| · φ · R · I" },
      { id: "ph", label: "Phase unwrap", sub: "atan2 + ±180° rule" },
      { id: "spk", label: "Spike fix", sub: "Hampel / median / local" },
      { id: "avg", label: "Complex average", sub: "optional repeated .s2p" },
      { id: "flt", label: "Mild filter", sub: "same settings on R, I", hub: true },
      { id: "ham", label: "Hamming copy", sub: "keep filtered too" },
      { id: "nrm", label: "NRMSE + report", sub: "validation summary", hub: true },
    ],
    edges: [
      { from: "in", to: "hdr" },
      { from: "hdr", to: "chk" },
      { from: "chk", to: "uni" },
      { from: "uni", to: "plot" },
      { from: "plot", to: "ph" },
      { from: "ph", to: "spk" },
      { from: "spk", to: "avg" },
      { from: "avg", to: "flt" },
      { from: "flt", to: "ham" },
      { from: "ham", to: "nrm" },
    ],
  },
  m2w3: {
    title: "Module 2 — Week 3 (time domain)",
    blurb: "Matched Hamming on target and reference, then IFFT. Clutter needs ≥2 channels (skipped on single .s2p).",
    direction: "vertical",
    nodes: [
      { id: "flt", label: "S21_filtered", sub: "target (+ ref filtered)" },
      { id: "win", label: "Matched Hamming", sub: "identical w_H", hub: true },
      { id: "ifft", label: "IFFT + time_s", sub: "Δt = 1/(N Δf)" },
      { id: "ref", label: "Time ref subtract", sub: "skip if no match" },
      { id: "clu", label: "Group-mean clutter", sub: "if ≥2 channels" },
      { id: "alpha", label: "Global normalize", sub: "α = max |s|", hub: true },
      { id: "meta", label: "module2_week3", sub: "metadata arrays" },
    ],
    edges: [
      { from: "flt", to: "win" },
      { from: "win", to: "ifft" },
      { from: "ifft", to: "ref" },
      { from: "ref", to: "clu" },
      { from: "clu", to: "alpha" },
      { from: "alpha", to: "meta" },
    ],
  },
  handoff: {
    title: "Module 3 handoff export",
    blurb: "GUI: Export Module 3 Handoff. Writes .mat + optional .csv + .json parameter report with Tx/Rx coordinates.",
    direction: "horizontal",
    nodes: [
      { id: "res", label: "PreprocessingResult", sub: "+ Week 3", hub: true },
      { id: "pay", label: "build_handover_payload", sub: "handover_export.py" },
      { id: "mat", label: ".mat file", sub: "frequency_Hz · time_s · S21_*" },
      { id: "csv", label: ".csv", sub: "|S21| dB vs f" },
      { id: "json", label: ".json params", sub: "filter · NRMSE · Week 3" },
      { id: "geo", label: "Tx / Rx coords", sub: "circle if missing" },
      { id: "m3", label: "Module 3 input", sub: "file ready; GUI TBD", hub: true },
    ],
    edges: [
      { from: "res", to: "pay" },
      { from: "pay", to: "mat" },
      { from: "pay", to: "csv" },
      { from: "pay", to: "json" },
      { from: "pay", to: "geo" },
      { from: "mat", to: "m3" },
      { from: "geo", to: "m3", dashed: true },
    ],
  },
};

function FlowDiagram({
  nodes,
  edges,
  direction,
}: {
  nodes: FlowNode[];
  edges: FlowEdge[];
  direction: "horizontal" | "vertical";
}) {
  const theme = useHostTheme();
  const nodeW = direction === "horizontal" ? 148 : 168;
  const nodeH = 52;
  const layout = computeDAGLayout({
    nodes: nodes.map((n) => ({ id: n.id })),
    edges: edges.map((e) => ({ from: e.from, to: e.to })),
    direction,
    nodeWidth: nodeW,
    nodeHeight: nodeH,
    rankGap: direction === "horizontal" ? 44 : 36,
    nodeGap: 24,
    padding: 12,
  });
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const dashByKey = Object.fromEntries(
    edges.map((e) => [`${e.from}->${e.to}`, Boolean(e.dashed)]),
  );

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${layout.width} ${layout.height}`}
      style={{ display: "block", maxWidth: layout.width }}
    >
      {layout.ranks.map((rank) => (
        <rect
          key={rank.rank}
          x={rank.x}
          y={rank.y}
          width={rank.width}
          height={rank.height}
          rx={6}
          fill={theme.fill.tertiary}
          opacity={0.4}
        />
      ))}
      {layout.edges.map((e, i) => (
        <line
          key={i}
          x1={e.sourceX}
          y1={e.sourceY}
          x2={e.targetX}
          y2={e.targetY}
          stroke={theme.stroke.secondary}
          strokeWidth={1.5}
          strokeDasharray={dashByKey[`${e.from}->${e.to}`] ? "4 3" : undefined}
        />
      ))}
      {layout.nodes.map((n) => {
        const meta = byId[n.id];
        return (
          <g key={n.id} transform={`translate(${n.x}, ${n.y})`}>
            <rect
              width={nodeW}
              height={nodeH}
              rx={6}
              fill={meta.hub ? theme.fill.secondary : theme.bg.elevated}
              stroke={meta.hub ? theme.accent.primary : theme.stroke.primary}
              strokeWidth={meta.hub ? 1.5 : 1}
            />
            <text
              x={nodeW / 2}
              y={22}
              textAnchor="middle"
              fill={theme.text.primary}
              fontSize={11}
              fontWeight={600}
            >
              {meta.label}
            </text>
            <text
              x={nodeW / 2}
              y={38}
              textAnchor="middle"
              fill={theme.text.tertiary}
              fontSize={10}
            >
              {meta.sub}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function Module1Module2Flowchart() {
  const [tab, setTab] = useState<TabId>("overview");
  const graph = GRAPHS[tab];

  return (
    <Stack gap={20} style={{ padding: 24, maxWidth: 1040 }}>
      <Stack gap={8}>
        <Row gap={8} wrap>
          <Pill active size="sm">
            Modules 1 + 2 complete
          </Pill>
          <Pill size="sm">UG S21 brief</Pill>
          <Pill size="sm">Week 3 handoff</Pill>
        </Row>
        <H1>Microwave framework flowchart</H1>
        <Text tone="secondary">
          Interactive view of Data Acquisition and Signal Preprocessing. Dashed
          edges are optional paths or not yet the default Module 3 GUI input.
        </Text>
      </Stack>

      <Row gap={16} wrap>
        <Stat value="Done" label="Module 1 — Acquisition" tone="success" />
        <Stat value="Done" label="Module 2 — Preprocessing" tone="success" />
        <Stat value="Open" label="Module 3 — consume handoff" tone="warning" />
      </Row>

      <Row gap={8} wrap>
        {TABS.map((t) => (
          <Pill key={t.id} active={tab === t.id} onClick={() => setTab(t.id)}>
            {t.label}
          </Pill>
        ))}
      </Row>

      <Stack gap={10}>
        <H2>{graph.title}</H2>
        <Text tone="secondary">{graph.blurb}</Text>
        <FlowDiagram nodes={graph.nodes} edges={graph.edges} direction={graph.direction} />
      </Stack>

      <Divider />

      <Callout tone="info" title="Where this lives in code">
        Module 1: data_loader/ + gui/upload_page.py. Module 2: preprocessing/
        touchstone_s21.py, week3_time_domain.py, handover_export.py, and the
        PreprocessingPanel in gui/main_window.py. Mermaid copies also sit in
        README §6.
      </Callout>
    </Stack>
  );
}
