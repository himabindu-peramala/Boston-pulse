import { useState, useEffect, useRef } from "react";
import BostonMap from "./components/BostonMap";

type Page = "home" | "navigate" | "chat" | "dashboard";
type SafetyLevel = "safe" | "moderate" | "caution";
interface Message { id: number; role: "user" | "assistant"; text: string; sources?: string[]; loading?: boolean; }
interface RouteResult { time: string; distance: string; safetyScore: number; level: SafetyLevel; notes: string[]; coordinates?: [number, number][]; }
interface SavedRoute { id: number; from: string; to: string; safety: SafetyLevel; time: string; date: string; }

const C = {
  bg: "#080b12", bgCard: "#0f1520", bgElevated: "#141c2e", bgHover: "#1a2236",
  border: "#1e2a40", borderMid: "#253350", borderLight: "#2e3d5c",
  text: "#e2e8f8", textMuted: "#6b7a99", textDim: "#3d4d6b",
  purple: "#6c5ce7", purpleBright: "#8b7ff0", purpleDim: "#1a1640",
  teal: "#00d4aa", amber: "#f59e0b", red: "#ef4444", green: "#10b981", blue: "#3b82f6",
};

const BASE_URL = (import.meta as any).env?.VITE_API_URL || "http://localhost:8000";

const api = {
  // ── Navigate: Purvaja's real API ─────────────────────────────────────────
  // NOTE: Ask Purvaja to confirm the request body field names before switching on
  navigate: async (from: string, to: string, weight: number): Promise<RouteResult> => {
    const res = await fetch(`${BASE_URL}/api/navigate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from, to, safety_weight: weight }),
    });
    const data = await res.json();

    // Pick route based on safety weight slider
    const preferred = weight > 66 ? "safest" : weight > 33 ? "balanced" : "fastest";
    const route = data.routes.find((r: any) => r.rank_label === preferred) ?? data.routes[0];

    // Map rank_label → SafetyLevel
    const level: SafetyLevel =
      route.rank_label === "safest"   ? "safe"     :
      route.rank_label === "balanced" ? "moderate" : "caution";

    // Convert units
    const mins  = Math.round(route.duration_s / 60);
    const miles = (route.distance_m / 1609.34).toFixed(1);

    const notes: Record<string, string[]> = {
      safe:     ["Safest available route selected", "Lower incident density corridor", "Well-lit streets prioritised"],
      moderate: ["Balanced route — speed vs safety", "Some incident zones avoided", "Good foot traffic coverage"],
      caution:  ["Fastest route selected", "Higher incident zones on path", "Consider safest route after midnight"],
    };

    return {
      time:        `${mins} min`,
      distance:    `${miles} mi`,
      safetyScore: Math.round(route.safety_score),
      level,
      notes:       notes[level],
      coordinates: route.geometry,
    };
  },

  // ── Chat: still mock — uncomment real call when Sushma's API is ready ────
  // chat: async (msg: string): Promise<{ text: string; sources: string[] }> => {
  //   const res = await fetch(`${BASE_URL}/api/chat`, {
  //     method: "POST",
  //     headers: { "Content-Type": "application/json" },
  //     body: JSON.stringify({ message: msg }),
  //   });
  //   return res.json();
  // },
  chat: async (msg: string): Promise<{ text: string; sources: string[] }> => {
    await new Promise(r => setTimeout(r, 1400));
    const safe = msg.toLowerCase().includes("safe");
    return safe
      ? { text: "Based on Boston's crime data for the past 90 days, that corridor has a safety score of 84/100 after 10 pm. There are no active incidents flagged in the past 48 hours.\n\nKey factors:\n• Well-lit streets with active streetlight coverage\n• High foot traffic from nearby T stops\n• Only 3 incidents reported in past 30 days (city avg: 11)\n\nI'd recommend staying on Huntington Ave rather than cutting through the park.", sources: ["BPD Crime Incident Reports", "Streetlight Locations dataset", "Vision Zero Safety Concerns"] }
      : { text: "Based on Boston's 311 data, pothole repairs in Mission Hill average 11.2 business days — about 40% longer than the city average of 7.9 days.\n\nFactors affecting your timeline:\n• Current queue depth: 847 open requests citywide\n• Winter/spring repairs take 2–3× longer\n• District crew availability: medium (2 active crews)\n\nYour request has a 73% probability of resolution within 14 days.", sources: ["Boston 311 Service Requests", "City Score dataset", "Analyze Boston"] };
  },
};

function SafetyBadge({ level, size = "sm" }: { level: SafetyLevel; size?: "sm" | "lg" }) {
  const m = { safe: [C.green, "#052e1c", "Safe"], moderate: [C.amber, "#2d1f04", "Moderate"], caution: [C.red, "#2d0808", "High risk"] };
  const [color, bg, label] = m[level];
  return (
    <span style={{ background: bg, color, border: `1px solid ${color}35`, borderRadius: 5, padding: size === "lg" ? "4px 12px" : "2px 8px", fontSize: size === "lg" ? 12 : 10, fontWeight: 600, display: "inline-flex", alignItems: "center", gap: 5 }}>
      <span style={{ width: 5, height: 5, borderRadius: "50%", background: color, display: "inline-block" }} />{label}
    </span>
  );
}

function PulseIndicator({ color = C.green }: { color?: string }) {
  return (
    <span style={{ position: "relative", display: "inline-flex", width: 8, height: 8, flexShrink: 0 }}>
      <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: color, opacity: 0.4, animation: "pulse-ring 1.5s ease-out infinite" }} />
      <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, position: "relative" }} />
    </span>
  );
}

function ScoreRing({ score, size = 72 }: { score: number; size?: number }) {
  const r = (size - 8) / 2, circ = 2 * Math.PI * r;
  const color = score >= 80 ? C.green : score >= 65 ? C.amber : C.red;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={C.border} strokeWidth={5} />
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth={5}
          strokeDasharray={circ} strokeDashoffset={circ * (1 - score / 100)}
          strokeLinecap="round" style={{ transition: "stroke-dashoffset 0.8s ease" }} />
        <text x={size/2} y={size/2} textAnchor="middle" dominantBaseline="central" fill={color}
          fontSize={size * 0.22} fontWeight={700}
          style={{ transform: "rotate(90deg)", transformOrigin: `${size/2}px ${size/2}px` }}>{score}</text>
      </svg>
      <span style={{ fontSize: 9, color: C.textMuted, letterSpacing: 0.8, textTransform: "uppercase" }}>Safety</span>
    </div>
  );
}

function AnimatedBg() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext("2d")!;
    let W = c.width = c.offsetWidth;
    let H = c.height = c.offsetHeight;
    let raf = 0;
    const orbs = [
      { x: W*0.15, y: H*0.4,  vx: 0.22,  vy: 0.12,  r: 340, color: C.purple+"16" },
      { x: W*0.78, y: H*0.3,  vx: -0.15, vy: 0.18,  r: 280, color: C.teal+"10"   },
      { x: W*0.5,  y: H*0.75, vx: 0.13,  vy: -0.16, r: 220, color: C.purple+"0c" },
    ];
    function draw() {
      ctx.clearRect(0, 0, W, H);
      orbs.forEach(o => {
        o.x += o.vx; o.y += o.vy;
        if (o.x < -o.r || o.x > W+o.r) o.vx *= -1;
        if (o.y < -o.r || o.y > H+o.r) o.vy *= -1;
        const g = ctx.createRadialGradient(o.x,o.y,0,o.x,o.y,o.r);
        g.addColorStop(0, o.color); g.addColorStop(1, "transparent");
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(o.x,o.y,o.r,0,Math.PI*2); ctx.fill();
      });
      raf = requestAnimationFrame(draw);
    }
    draw();
    const onResize = () => { W = c.width = c.offsetWidth; H = c.height = c.offsetHeight; };
    window.addEventListener("resize", onResize);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", onResize); };
  }, []);
  return <canvas ref={ref} style={{ position:"absolute", inset:0, width:"100%", height:"100%", pointerEvents:"none" }} />;
}

function NavBar({ page, setPage }: { page: Page; setPage: (p: Page) => void }) {
  const tabs = [
    { id: "home"      as Page, label: "Home",      icon: "⌂" },
    { id: "navigate"  as Page, label: "Navigate",  icon: "◎" },
    { id: "chat"      as Page, label: "Chat",      icon: "✦" },
    { id: "dashboard" as Page, label: "Dashboard", icon: "▦" },
  ];
  return (
    <nav style={{ background:`${C.bgCard}f8`, backdropFilter:"blur(12px)", borderBottom:`1px solid ${C.border}`, display:"flex", alignItems:"center", padding:"0 28px", height:56, position:"sticky", top:0, zIndex:200, width:"100%" }}>
      <div onClick={() => setPage("home")} style={{ display:"flex", alignItems:"center", gap:10, marginRight:32, cursor:"pointer" }}>
        <div style={{ width:30, height:30, borderRadius:8, background:`linear-gradient(135deg,${C.purple},${C.teal})`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:14, boxShadow:`0 0 16px ${C.purple}50` }}>◉</div>
        <div>
          <div style={{ color:C.text, fontWeight:700, fontSize:14, letterSpacing:0.2 }}>Boston Pulse</div>
          <div style={{ color:C.textMuted, fontSize:9, letterSpacing:1, textTransform:"uppercase" }}>Civic Intelligence</div>
        </div>
      </div>
      <div style={{ display:"flex", gap:2, flex:1 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setPage(t.id)}
            style={{ background:page===t.id?C.purpleDim:"transparent", color:page===t.id?C.purpleBright:C.textMuted, border:page===t.id?`1px solid ${C.purple}50`:"1px solid transparent", borderRadius:8, padding:"6px 16px", fontSize:12, fontWeight:page===t.id?600:400, cursor:"pointer", display:"flex", alignItems:"center", gap:6, transition:"all 0.15s" }}
            onMouseEnter={e => { if (page!==t.id) { const b=e.currentTarget as HTMLButtonElement; b.style.color=C.text; b.style.background=C.bgHover; } }}
            onMouseLeave={e => { if (page!==t.id) { const b=e.currentTarget as HTMLButtonElement; b.style.color=C.textMuted; b.style.background="transparent"; } }}>
            <span style={{ fontSize:10 }}>{t.icon}</span>{t.label}
          </button>
        ))}
      </div>
      <div style={{ display:"flex", alignItems:"center", gap:12 }}>
        <div style={{ display:"flex", alignItems:"center", gap:6, background:C.bgElevated, border:`1px solid ${C.border}`, borderRadius:20, padding:"4px 12px 4px 8px" }}>
          <PulseIndicator color={C.green} />
          <span style={{ fontSize:10, color:C.textMuted }}>Live</span>
        </div>
        <div style={{ width:32, height:32, borderRadius:"50%", background:`linear-gradient(135deg,${C.purple},${C.purpleBright})`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:11, color:"#fff", fontWeight:700, border:`2px solid ${C.purpleBright}40`, cursor:"pointer" }}>SG</div>
      </div>
    </nav>
  );
}

function HomePage({ setPage }: { setPage: (p: Page) => void }) {
  const [q, setQ] = useState("");
  const features = [
    { icon:"◎", title:"Safety-first navigation", sub:"XGBoost · 530K+ incidents", desc:"Routes that adapt to time-of-day risk, lighting data, and real-time incident feeds from Boston Police.", page:"navigate" as Page, color:C.teal   },
    { icon:"✦", title:"Civic intelligence chat",  sub:"LLM + RAG pipeline",        desc:"Ask anything about Boston in plain English. Every response cites its source — no hallucinations.",      page:"chat"      as Page, color:C.purple },
    { icon:"▦", title:"Your dashboard",           sub:"History · preferences",      desc:"Saved routes, query history, and safety preferences. Your personal civic data hub.",                      page:"dashboard" as Page, color:C.amber  },
  ];
  return (
    <div style={{ width:"100%", background:C.bg }}>
      <div style={{ width:"100%", position:"relative", overflow:"hidden", borderBottom:`1px solid ${C.border}`, padding:"88px 0 72px" }}>
        <AnimatedBg />
        <div style={{ position:"relative", maxWidth:720, margin:"0 auto", textAlign:"center", padding:"0 40px" }}>
          <div style={{ display:"inline-flex", alignItems:"center", gap:8, background:C.bgElevated, border:`1px solid ${C.borderMid}`, borderRadius:20, padding:"5px 14px 5px 8px", marginBottom:28 }}>
            <PulseIndicator color={C.teal} />
            <span style={{ fontSize:11, color:C.textMuted }}>Live · 6 data sources · updated 12 min ago</span>
          </div>
          <h1 style={{ color:C.text, fontSize:52, fontWeight:800, margin:"0 0 16px", lineHeight:1.12, letterSpacing:-1.5 }}>
            The Bostonian that knows<br />
            <span style={{ background:`linear-gradient(135deg,${C.purple},${C.teal})`, WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>the city inside out</span>
          </h1>
          <p style={{ color:C.textMuted, fontSize:15, margin:"0 auto 36px", lineHeight:1.8, maxWidth:500 }}>
            ML-driven civic intelligence — safe routing, 311 timelines, and neighborhood insights in one conversational platform.
          </p>
          <div style={{ display:"flex", gap:10, maxWidth:580, margin:"0 auto" }}>
            <div style={{ flex:1, position:"relative" }}>
              <span style={{ position:"absolute", left:14, top:"50%", transform:"translateY(-50%)", color:C.textDim, fontSize:13 }}>✦</span>
              <input value={q} onChange={e => setQ(e.target.value)} onKeyDown={e => e.key==="Enter" && setPage("chat")}
                placeholder='Try "Is Roxbury safe to walk at midnight?"'
                style={{ width:"100%", background:`${C.bgCard}cc`, border:`1px solid ${C.borderMid}`, borderRadius:12, padding:"13px 14px 13px 36px", fontSize:13, color:C.text, outline:"none", boxSizing:"border-box", backdropFilter:"blur(8px)" }}
                onFocus={e => (e.target as HTMLInputElement).style.borderColor=C.purple}
                onBlur={e  => (e.target as HTMLInputElement).style.borderColor=C.borderMid} />
            </div>
            <button onClick={() => setPage("chat")} style={{ background:`linear-gradient(135deg,${C.purple},${C.purpleBright})`, border:"none", borderRadius:12, padding:"13px 26px", color:"#fff", fontSize:13, fontWeight:700, cursor:"pointer", whiteSpace:"nowrap", boxShadow:`0 4px 20px ${C.purple}40` }}>Ask →</button>
          </div>
        </div>
      </div>
      <div style={{ width:"100%", display:"grid", gridTemplateColumns:"1fr 1fr 1fr", borderBottom:`1px solid ${C.border}` }}>
        {features.map((f,i) => (
          <div key={f.title} onClick={() => setPage(f.page)}
            style={{ padding:"32px 36px", cursor:"pointer", transition:"background 0.2s", borderRight:i<2?`1px solid ${C.border}`:"none", background:"transparent" }}
            onMouseEnter={e => (e.currentTarget as HTMLDivElement).style.background=C.bgCard}
            onMouseLeave={e => (e.currentTarget as HTMLDivElement).style.background="transparent"}>
            <div style={{ width:42, height:42, borderRadius:10, background:`${f.color}15`, border:`1px solid ${f.color}25`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:18, color:f.color, marginBottom:16 }}>{f.icon}</div>
            <div style={{ color:C.text, fontWeight:700, fontSize:14, marginBottom:4 }}>{f.title}</div>
            <div style={{ fontSize:10, color:f.color, marginBottom:10, fontWeight:500, letterSpacing:0.3 }}>{f.sub}</div>
            <div style={{ color:C.textMuted, fontSize:12, lineHeight:1.7, marginBottom:18 }}>{f.desc}</div>
            <div style={{ fontSize:11, color:C.textMuted }}>Open →</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function NavigatePage() {
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [weight, setWeight] = useState(70);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RouteResult | null>(null);
  const [activeLayer, setActiveLayer] = useState("safety");
  const [error, setError] = useState<string | null>(null);

  async function compute() {
    if (!from || !to) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const r = await api.navigate(from, to, weight);
      setResult(r);
    } catch (err) {
      console.error("Navigate API error:", err);
      setError("Could not compute route. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  }

  const quick = [["Northeastern → Back Bay T","safe"],["Mission Hill → Fenway","moderate"],["Roxbury → Downtown","caution"],["North End → South Station","safe"]];

  return (
    <div style={{ display:"grid", gridTemplateColumns:"360px 1fr", height:"calc(100vh - 56px)", width:"100%" }}>
      <div style={{ background:C.bgCard, borderRight:`1px solid ${C.border}`, overflowY:"auto", display:"flex", flexDirection:"column" }}>
        <div style={{ padding:"18px 20px 14px", borderBottom:`1px solid ${C.border}` }}>
          <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:3 }}>
            <span style={{ color:C.teal, fontSize:15 }}>◎</span>
            <span style={{ color:C.text, fontWeight:700, fontSize:14 }}>Safety Navigation</span>
          </div>
          <div style={{ fontSize:11, color:C.textMuted }}>XGBoost model · 530K+ incidents</div>
        </div>
        <div style={{ padding:20, display:"flex", flexDirection:"column", gap:16, flex:1 }}>
          <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
            {([["A",C.teal,from,setFrom,"Origin — e.g. Northeastern"],["B",C.red,to,setTo,"Destination — e.g. Back Bay T"]] as const).map(([lbl,col,val,set,ph]) => (
              <div key={lbl} style={{ position:"relative" }}>
                <div style={{ position:"absolute", left:11, top:"50%", transform:"translateY(-50%)", width:18, height:18, borderRadius:"50%", background:`${col}18`, border:`1px solid ${col}50`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:9, color:col, fontWeight:700 }}>{lbl}</div>
                <input value={val} onChange={e=>set(e.target.value)} onKeyDown={e=>e.key==="Enter"&&compute()} placeholder={ph}
                  style={{ width:"100%", background:C.bgElevated, border:`1px solid ${C.border}`, borderRadius:10, padding:"10px 12px 10px 36px", fontSize:12, color:C.text, outline:"none", boxSizing:"border-box" }}
                  onFocus={e=>(e.target as HTMLInputElement).style.borderColor=col}
                  onBlur={e=>(e.target as HTMLInputElement).style.borderColor=C.border} />
              </div>
            ))}
          </div>
          <div style={{ background:C.bgElevated, border:`1px solid ${C.border}`, borderRadius:10, padding:"12px 14px" }}>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom:8 }}>
              <span style={{ fontSize:11, color:C.textMuted, fontWeight:600, textTransform:"uppercase", letterSpacing:0.8 }}>Safety weight</span>
              <span style={{ fontSize:12, color:C.purple, fontWeight:700 }}>{weight}%</span>
            </div>
            <input type="range" min={0} max={100} value={weight} onChange={e=>setWeight(+e.target.value)} style={{ width:"100%", accentColor:C.purple, marginBottom:4 }} />
            <div style={{ display:"flex", justifyContent:"space-between", fontSize:10, color:C.textDim }}>
              <span>Fastest</span><span>Safest</span>
            </div>
          </div>
          <button onClick={compute} disabled={!from||!to||loading}
            style={{ background:from&&to?`linear-gradient(135deg,${C.purple},${C.purpleBright})`:C.bgHover, border:`1px solid ${from&&to?C.purple:C.border}`, borderRadius:10, padding:"11px", color:from&&to?"#fff":C.textDim, fontSize:13, fontWeight:700, cursor:from&&to?"pointer":"default", transition:"all 0.2s", boxShadow:from&&to?`0 4px 16px ${C.purple}35`:"none" }}>
            {loading ? "Computing…" : "Find safe route →"}
          </button>

          {error && (
            <div style={{ background:"#2d0808", border:`1px solid ${C.red}40`, borderRadius:8, padding:"10px 12px", fontSize:11, color:C.red }}>
              {error}
            </div>
          )}

          {loading && (
            <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
              <div style={{ fontSize:11, color:C.textMuted }}>Analyzing incidents…</div>
              {[100,75,90].map((w,i) => <div key={i} style={{ height:12, borderRadius:4, background:C.bgHover, width:`${w}%` }} />)}
            </div>
          )}
          {result && !loading && (
            <div style={{ background:C.bg, border:`1px solid ${C.borderMid}`, borderRadius:12, overflow:"hidden" }}>
              <div style={{ padding:"12px 14px", borderBottom:`1px solid ${C.border}`, display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                <span style={{ color:C.text, fontWeight:600, fontSize:13 }}>Route found</span>
                <SafetyBadge level={result.level} size="lg" />
              </div>
              <div style={{ padding:"14px" }}>
                <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:14 }}>
                  <div style={{ display:"flex", gap:14 }}>
                    <div><div style={{ color:C.text, fontSize:18, fontWeight:700 }}>{result.time}</div><div style={{ color:C.textMuted, fontSize:10, marginTop:2 }}>travel time</div></div>
                    <div style={{ width:1, background:C.border }} />
                    <div><div style={{ color:C.text, fontSize:18, fontWeight:700 }}>{result.distance}</div><div style={{ color:C.textMuted, fontSize:10, marginTop:2 }}>distance</div></div>
                  </div>
                  <ScoreRing score={result.safetyScore} />
                </div>
                <div style={{ height:1, background:C.border, marginBottom:12 }} />
                {result.notes.map((n,i) => (
                  <div key={i} style={{ display:"flex", gap:8, fontSize:12, color:C.textMuted, marginBottom:7, lineHeight:1.5 }}>
                    <span style={{ color:result.level==="safe"?C.teal:result.level==="moderate"?C.amber:C.red, flexShrink:0 }}>—</span>{n}
                  </div>
                ))}
              </div>
            </div>
          )}
          <div style={{ height:1, background:C.border }} />
          <div>
            <div style={{ fontSize:10, color:C.textMuted, fontWeight:600, letterSpacing:1, marginBottom:8, textTransform:"uppercase" }}>Common routes</div>
            <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
              {quick.map(([r,l]) => (
                <div key={r} onClick={() => { const [f,t]=r.split(" → "); setFrom(f); setTo(t); }}
                  style={{ display:"flex", justifyContent:"space-between", alignItems:"center", padding:"8px 10px", borderRadius:8, cursor:"pointer", background:C.bgElevated, border:`1px solid ${C.border}`, transition:"border-color 0.15s" }}
                  onMouseEnter={e=>(e.currentTarget as HTMLDivElement).style.borderColor=C.borderLight}
                  onMouseLeave={e=>(e.currentTarget as HTMLDivElement).style.borderColor=C.border}>
                  <span style={{ fontSize:11, color:C.textMuted }}>{r}</span>
                  <SafetyBadge level={l as SafetyLevel} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      <div style={{ position:"relative" }}>
        <BostonMap result={result} loading={loading} />
        <div style={{ position:"absolute", top:14, left:14, display:"flex", flexDirection:"column", gap:5 }}>
          {[["safety","Safety heatmap"],["crime","Crime density"],["light","Lighting"],["transit","MBTA stops"]].map(([id,label]) => (
            <button key={id} onClick={() => setActiveLayer(id)}
              style={{ background:activeLayer===id?`${C.purple}cc`:C.bgCard+"cc", border:`1px solid ${activeLayer===id?C.purple:C.border}`, borderRadius:7, padding:"5px 11px", fontSize:11, color:activeLayer===id?C.purpleBright:C.textMuted, cursor:"pointer", backdropFilter:"blur(8px)", textAlign:"left" }}>
              {label}
            </button>
          ))}
        </div>
        <div style={{ position:"absolute", top:14, right:14, background:`${C.bgCard}ee`, border:`1px solid ${C.border}`, borderRadius:9, padding:"10px 13px", backdropFilter:"blur(8px)" }}>
          <div style={{ fontSize:9, color:C.textDim, textTransform:"uppercase", letterSpacing:1, marginBottom:7, fontWeight:600 }}>Risk level</div>
          {[[C.red,"High risk"],[C.amber,"Moderate"],[C.green,"Low risk"]].map(([c,l]) => (
            <div key={l as string} style={{ display:"flex", alignItems:"center", gap:7, marginBottom:5 }}>
              <div style={{ width:8, height:8, borderRadius:"50%", background:c as string }} />
              <span style={{ fontSize:11, color:C.textMuted }}>{l}</span>
            </div>
          ))}
        </div>
        {result && (
          <div style={{ position:"absolute", bottom:0, left:0, right:0, background:`${C.bgCard}f0`, borderTop:`1px solid ${C.border}`, padding:"9px 18px", display:"flex", alignItems:"center", gap:14, backdropFilter:"blur(8px)" }}>
            <PulseIndicator color={result.level==="safe"?C.green:result.level==="moderate"?C.amber:C.red} />
            <span style={{ fontSize:12, color:C.text, fontWeight:600 }}>Route active</span>
            <span style={{ fontSize:11, color:C.textMuted }}>{from} → {to}</span>
            <span style={{ fontSize:11, color:C.textMuted, marginLeft:"auto" }}>XGBoost · last retrained 6h ago</span>
          </div>
        )}
      </div>
    </div>
  );
}

function ChatPage() {
  const [msgs, setMsgs] = useState<Message[]>([
    { id:0, role:"assistant", text:"Hi — I'm Boston Pulse. Ask me anything about the city: safe routes, 311 timelines, neighborhood comparisons, permits, housing. Every response cites its sources.", sources:["Boston Open Data Portal"] }
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);
  const suggestions = [
    { icon:"◎", text:"Is it safe to walk from Northeastern to Fenway at midnight?" },
    { icon:"⏱", text:"When will my 311 pothole request be resolved?" },
    { icon:"🏠", text:"Which neighborhoods suit a grad student on $1,800/mo?" },
    { icon:"📋", text:"What permits do I need for an outdoor event?" },
  ];
  async function send(text = input) {
    if (!text.trim() || busy) return;
    setMsgs(p => [...p, { id:Date.now(), role:"user", text }, { id:Date.now()+1, role:"assistant", text:"", loading:true }]);
    setInput("");
    setBusy(true);
    try {
      const r = await api.chat(text);
      setMsgs(p => p.map(m => m.loading ? { ...m, loading:false, text:r.text, sources:r.sources } : m));
    } catch {
      setMsgs(p => p.map(m => m.loading ? { ...m, loading:false, text:"Something went wrong. Please try again." } : m));
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => { bottom.current?.scrollIntoView({ behavior:"smooth" }); }, [msgs]);
  return (
    <div style={{ display:"flex", height:"calc(100vh - 56px)", width:"100%", background:C.bg }}>
      <div style={{ width:240, background:C.bgCard, borderRight:`1px solid ${C.border}`, display:"flex", flexDirection:"column", padding:16, gap:10, flexShrink:0 }}>
        <div style={{ fontSize:10, color:C.textMuted, fontWeight:600, letterSpacing:1, textTransform:"uppercase", marginBottom:2 }}>Active data sources</div>
        {[["Crime Reports","530K+ records"],["Service 311","500K+ requests"],["Food Inspections","868K+ records"],["BERDO Emissions","5K+ buildings"],["City Score","65K+ metrics"],["Street Sweeping","Schedule data"]].map(([name,sub]) => (
          <div key={name} style={{ display:"flex", justifyContent:"space-between", alignItems:"center", background:C.bgElevated, border:`1px solid ${C.border}`, borderRadius:8, padding:"8px 10px" }}>
            <div>
              <div style={{ fontSize:11, color:C.text, fontWeight:500 }}>{name}</div>
              <div style={{ fontSize:9, color:C.textDim, marginTop:1 }}>{sub}</div>
            </div>
            <PulseIndicator color={C.green} />
          </div>
        ))}
        <div style={{ marginTop:"auto", background:C.bgElevated, border:`1px solid ${C.border}`, borderRadius:8, padding:"10px 12px" }}>
          <div style={{ fontSize:9, color:C.textDim, textTransform:"uppercase", letterSpacing:1, marginBottom:6 }}>RAG pipeline</div>
          {["Grounded responses","Source citations","Hallucination guard"].map(f => (
            <div key={f} style={{ fontSize:11, color:C.textMuted, marginBottom:3 }}>✓ {f}</div>
          ))}
        </div>
      </div>
      <div style={{ flex:1, display:"flex", flexDirection:"column", overflow:"hidden" }}>
        <div style={{ flex:1, overflowY:"auto", padding:"28px 0" }}>
          <div style={{ maxWidth:720, margin:"0 auto", padding:"0 24px", display:"flex", flexDirection:"column", gap:20 }}>
            {msgs.map(m => (
              <div key={m.id} style={{ display:"flex", flexDirection:"column", alignItems:m.role==="user"?"flex-end":"flex-start" }}>
                {m.role==="assistant" && (
                  <div style={{ display:"flex", gap:10, alignItems:"flex-start", maxWidth:"88%" }}>
                    <div style={{ width:28, height:28, borderRadius:8, background:`linear-gradient(135deg,${C.purple},${C.teal})`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:11, flexShrink:0, marginTop:2 }}>◉</div>
                    <div>
                      <div style={{ fontSize:10, color:C.textDim, marginBottom:5 }}>Boston Pulse</div>
                      <div style={{ background:C.bgCard, border:`1px solid ${C.border}`, borderRadius:"3px 12px 12px 12px", padding:"13px 16px", fontSize:13, color:C.text, lineHeight:1.75, whiteSpace:"pre-line" }}>
                        {m.loading
                          ? <div style={{ display:"flex", gap:4, alignItems:"center", height:20 }}>{[0,1,2].map(i=><div key={i} style={{ width:6,height:6,borderRadius:"50%",background:C.purple,animation:`bounce 1.2s ${i*0.2}s infinite` }}/>)}</div>
                          : m.text}
                      </div>
                      {m.sources && !m.loading && (
                        <div style={{ display:"flex", gap:5, flexWrap:"wrap", marginTop:7 }}>
                          {m.sources.map(s => <span key={s} style={{ fontSize:10, color:C.textDim, background:C.bgElevated, border:`1px solid ${C.border}`, borderRadius:4, padding:"2px 7px" }}>📎 {s}</span>)}
                        </div>
                      )}
                    </div>
                  </div>
                )}
                {m.role==="user" && (
                  <div style={{ background:C.bgElevated, border:`1px solid ${C.borderMid}`, borderRadius:"12px 3px 12px 12px", padding:"11px 16px", fontSize:13, color:C.text, maxWidth:"70%", lineHeight:1.65 }}>{m.text}</div>
                )}
              </div>
            ))}
            <div ref={bottom} />
          </div>
        </div>
        {msgs.length===1 && (
          <div style={{ maxWidth:720, width:"100%", margin:"0 auto", padding:"0 24px 14px" }}>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:7 }}>
              {suggestions.map(s => (
                <button key={s.text} onClick={() => send(s.text)}
                  style={{ background:C.bgCard, border:`1px solid ${C.border}`, borderRadius:9, padding:"10px 13px", fontSize:12, color:C.textMuted, cursor:"pointer", textAlign:"left", lineHeight:1.5, display:"flex", gap:9, alignItems:"flex-start", transition:"all 0.15s" }}
                  onMouseEnter={e=>{const b=e.currentTarget;b.style.borderColor=C.purple;b.style.color=C.text;}}
                  onMouseLeave={e=>{const b=e.currentTarget;b.style.borderColor=C.border;b.style.color=C.textMuted;}}>
                  <span style={{ fontSize:13 }}>{s.icon}</span><span>{s.text}</span>
                </button>
              ))}
            </div>
          </div>
        )}
        <div style={{ background:C.bgCard, borderTop:`1px solid ${C.border}`, padding:"13px 24px" }}>
          <div style={{ maxWidth:720, margin:"0 auto", display:"flex", gap:8 }}>
            <input value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&!e.shiftKey&&send()}
              placeholder="Ask about safety, 311, housing, permits…"
              style={{ flex:1, background:C.bgElevated, border:`1px solid ${C.border}`, borderRadius:9, padding:"10px 14px", fontSize:13, color:C.text, outline:"none" }}
              onFocus={e=>(e.target as HTMLInputElement).style.borderColor=C.purple}
              onBlur={e=>(e.target as HTMLInputElement).style.borderColor=C.border} />
            <button onClick={()=>send()} disabled={!input.trim()||busy}
              style={{ background:input.trim()?`linear-gradient(135deg,${C.purple},${C.purpleBright})`:C.bgHover, border:"none", borderRadius:9, padding:"10px 20px", color:input.trim()?"#fff":C.textDim, fontSize:13, fontWeight:600, cursor:input.trim()?"pointer":"default" }}>
              Send
            </button>
          </div>
          <div style={{ maxWidth:720, margin:"5px auto 0", fontSize:10, color:C.textDim }}>Grounded in Boston Open Data. Verify safety-critical info independently.</div>
        </div>
      </div>
    </div>
  );
}

function DashboardPage() {
  const [weight, setWeight] = useState(70);
  const [tod, setTod] = useState("night");
  const routes: SavedRoute[] = [
    { id:1, from:"Northeastern University", to:"Back Bay Station", safety:"safe",     time:"14 min", date:"Today, 9:12 PM" },
    { id:2, from:"Mission Hill",            to:"Fenway Park",       safety:"moderate", time:"18 min", date:"Yesterday"       },
    { id:3, from:"Roxbury Crossing",        to:"Downtown Crossing", safety:"caution",  time:"22 min", date:"Mar 22"          },
  ];
  const history = [
    { q:"Is Tremont St safe after 11pm?",                     time:"2h ago",     tag:"Safety"  },
    { q:"311 resolution time for potholes in Mission Hill",   time:"Yesterday",  tag:"311"     },
    { q:"Accessible apartments near Green Line under $1,800", time:"2 days ago", tag:"Housing" },
    { q:"Outdoor event permit requirements Boston",           time:"3 days ago", tag:"Permits" },
  ];
  const tagColor: Record<string,string> = { Safety:C.red, "311":C.amber, Housing:C.blue, Permits:C.green };
  return (
    <div style={{ width:"100%", minHeight:"calc(100vh - 56px)", background:C.bg, padding:"28px 40px" }}>
      <div style={{ maxWidth:960, margin:"0 auto", display:"flex", flexDirection:"column", gap:16 }}>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
          <div style={{ background:C.bgCard, border:`1px solid ${C.border}`, borderRadius:12, overflow:"hidden" }}>
            <div style={{ padding:"14px 18px", borderBottom:`1px solid ${C.border}`, display:"flex", justifyContent:"space-between", alignItems:"center" }}>
              <span style={{ fontSize:13, color:C.text, fontWeight:600 }}>Saved routes</span>
              <span style={{ fontSize:11, color:C.purple, cursor:"pointer" }}>+ New</span>
            </div>
            {routes.map((r,i) => (
              <div key={r.id} style={{ padding:"12px 18px", borderBottom:i<routes.length-1?`1px solid ${C.border}`:"none", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                <div>
                  <div style={{ fontSize:12, color:C.text, fontWeight:500 }}>{r.from} → {r.to}</div>
                  <div style={{ fontSize:11, color:C.textMuted, marginTop:2 }}>{r.date} · {r.time}</div>
                </div>
                <SafetyBadge level={r.safety} size="lg" />
              </div>
            ))}
          </div>
          <div style={{ background:C.bgCard, border:`1px solid ${C.border}`, borderRadius:12, overflow:"hidden" }}>
            <div style={{ padding:"14px 18px", borderBottom:`1px solid ${C.border}`, display:"flex", justifyContent:"space-between" }}>
              <span style={{ fontSize:13, color:C.text, fontWeight:600 }}>Recent queries</span>
              <span style={{ fontSize:11, color:C.textMuted }}>View all</span>
            </div>
            {history.map((h,i) => (
              <div key={i} style={{ padding:"11px 18px", borderBottom:i<history.length-1?`1px solid ${C.border}`:"none", display:"flex", gap:10, alignItems:"flex-start" }}>
                <span style={{ fontSize:10, color:tagColor[h.tag], background:`${tagColor[h.tag]}15`, border:`1px solid ${tagColor[h.tag]}28`, borderRadius:4, padding:"2px 6px", flexShrink:0, marginTop:1, fontWeight:500 }}>{h.tag}</span>
                <div style={{ flex:1 }}>
                  <div style={{ fontSize:12, color:C.text, lineHeight:1.5 }}>{h.q}</div>
                  <div style={{ fontSize:10, color:C.textDim, marginTop:2 }}>{h.time}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div style={{ background:C.bgCard, border:`1px solid ${C.border}`, borderRadius:12, padding:"18px 20px" }}>
          <div style={{ fontSize:13, color:C.text, fontWeight:600, marginBottom:16 }}>Preferences</div>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:24 }}>
            <div>
              <div style={{ display:"flex", justifyContent:"space-between", marginBottom:8 }}>
                <span style={{ fontSize:12, color:C.textMuted }}>Safety vs speed</span>
                <span style={{ fontSize:12, color:C.purple, fontWeight:600 }}>{weight}%</span>
              </div>
              <input type="range" min={0} max={100} value={weight} onChange={e=>setWeight(+e.target.value)} style={{ width:"100%", accentColor:C.purple }} />
              <div style={{ display:"flex", justifyContent:"space-between", fontSize:10, color:C.textDim, marginTop:4 }}>
                <span>Fastest</span><span>Safest</span>
              </div>
            </div>
            <div>
              <div style={{ fontSize:12, color:C.textMuted, marginBottom:8 }}>Default time of day</div>
              <div style={{ display:"flex", gap:7 }}>
                {["morning","afternoon","night"].map(t => (
                  <button key={t} onClick={()=>setTod(t)}
                    style={{ flex:1, background:tod===t?C.purpleDim:C.bgElevated, border:`1px solid ${tod===t?C.purple:C.border}`, borderRadius:8, padding:"7px 0", fontSize:12, color:tod===t?C.purpleBright:C.textMuted, cursor:"pointer", fontWeight:tod===t?600:400, textTransform:"capitalize" }}>
                    {t}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("home");
  return (
    <div style={{ background:C.bg, minHeight:"100vh", width:"100%", fontFamily:"-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", color:C.text, overflowX:"hidden" }}>
      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html, body, #root { width: 100%; margin: 0; padding: 0; background: #080b12; overflow-x: hidden; }
        input::placeholder { color: #3d4d6b; }
        @keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-5px)} }
        @keyframes pulse-ring { 0%{transform:scale(1);opacity:0.4} 100%{transform:scale(2.5);opacity:0} }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #1e2a40; border-radius: 2px; }
      `}</style>
      <NavBar page={page} setPage={setPage} />
      {page==="home"      && <HomePage setPage={setPage} />}
      {page==="navigate"  && <NavigatePage />}
      {page==="chat"      && <ChatPage />}
      {page==="dashboard" && <DashboardPage />}
    </div>
  );
}