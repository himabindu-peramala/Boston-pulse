import { useEffect, useRef } from "react";
import type { Map, TileLayer, Polyline, CircleMarker, Circle } from "leaflet";

type SafetyLevel = "safe" | "moderate" | "caution";

interface RouteResult {
  time: string;
  distance: string;
  safetyScore: number;
  level: SafetyLevel;
  notes: string[];
  coordinates?: [number, number][];   // [[lat, lng], ...] from backend
}

interface BostonMapProps {
  result: RouteResult | null;
  loading: boolean;
}

// Risk zones overlaid on Boston (static for now — replace with real API data later)
const RISK_ZONES = [
  { lat: 42.3295, lng: -71.0832, radius: 300, color: "#ef4444", label: "High risk" },  // Roxbury
  { lat: 42.3354, lng: -71.0747, radius: 250, color: "#ef4444", label: "High risk" },  // S End edge
  { lat: 42.3519, lng: -71.0686, radius: 200, color: "#f59e0b", label: "Moderate"  },  // Back Bay
  { lat: 42.3467, lng: -71.0972, radius: 280, color: "#f59e0b", label: "Moderate"  },  // Allston
  { lat: 42.3398, lng: -71.0892, radius: 350, color: "#10b981", label: "Low risk"  },  // Fenway
  { lat: 42.3601, lng: -71.0589, radius: 300, color: "#10b981", label: "Low risk"  },  // Downtown
];

// Demo route coordinates (Northeastern → Back Bay T)
// Replace with real coordinates from api.navigate response
const DEMO_ROUTE: [number, number][] = [
  [42.3398, -71.0892],
  [42.3421, -71.0867],
  [42.3445, -71.0843],
  [42.3468, -71.0812],
  [42.3490, -71.0788],
  [42.3501, -71.0771],
];

export default function BostonMap({ result, loading }: BostonMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef       = useRef<Map | null>(null);
  const routeRef     = useRef<Polyline | null>(null);
  const markersRef   = useRef<CircleMarker[]>([]);
  const zonesRef     = useRef<Circle[]>([]);

  // ── Init map once ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    // Dynamic import so Leaflet doesn't break SSR / Vite
    import("leaflet").then(L => {
      // Fix default marker icon paths broken by Vite bundling
      delete (L.Icon.Default.prototype as any)._getIconUrl;
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
        iconUrl:       "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
        shadowUrl:     "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
      });

      const map = L.map(containerRef.current!, {
        center: [42.3398, -71.0892],   // Boston — Northeastern
        zoom: 14,
        zoomControl: true,
        attributionControl: false,
      });

      // CartoDB dark tiles — free, no API key
      L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        { maxZoom: 19 }
      ).addTo(map);

      // Draw risk zones
      RISK_ZONES.forEach(z => {
        const circle = L.circle([z.lat, z.lng], {
          radius: z.radius,
          color: z.color,
          fillColor: z.color,
          fillOpacity: 0.06,
          weight: 0.5,
          opacity: 0.25,
        }).addTo(map);
        zonesRef.current.push(circle);
      });

      mapRef.current = map;
    });

    return () => {
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, []);

  // ── Update route when result changes ──────────────────────────────────
  useEffect(() => {
    if (!mapRef.current) return;

    import("leaflet").then(L => {
      const map = mapRef.current!;

      // Clear previous route + markers
      routeRef.current?.remove();
      markersRef.current.forEach(m => m.remove());
      markersRef.current = [];

      if (!result || loading) return;

      const coords: [number, number][] = result.coordinates?.length
        ? result.coordinates
        : DEMO_ROUTE;

      const color =
        result.level === "safe"     ? "#00d4aa" :
        result.level === "moderate" ? "#f59e0b" : "#ef4444";

      // Glow effect — wide transparent line underneath
      L.polyline(coords, {
        color,
        weight: 12,
        opacity: 0.2,
      }).addTo(map);

      // Main route line
      routeRef.current = L.polyline(coords, {
        color,
        weight: 3,
        opacity: 0.9,
        dashArray: result.level === "caution" ? "8 5" : undefined,
      }).addTo(map);

      // Origin marker (A)
      const origin = coords[0];
      const dest   = coords[coords.length - 1];

      const markerA = L.circleMarker(origin, {
        radius: 8,
        color: "#00d4aa",
        fillColor: "#00d4aa",
        fillOpacity: 1,
        weight: 2,
      }).addTo(map).bindTooltip("A — Origin", { permanent: false });

      const markerB = L.circleMarker(dest, {
        radius: 8,
        color: "#ef4444",
        fillColor: "#ef4444",
        fillOpacity: 1,
        weight: 2,
      }).addTo(map).bindTooltip("B — Destination", { permanent: false });

      markersRef.current = [markerA, markerB];

      // Fit map to route
      map.fitBounds(L.latLngBounds(coords), { padding: [60, 60] });
    });
  }, [result, loading]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      {/* Leaflet needs a CSS import — add to index.css or here via link */}
      <link
        rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"
      />
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />

      {loading && (
        <div style={{
          position: "absolute", inset: 0,
          background: "rgba(8,11,18,0.6)",
          display: "flex", alignItems: "center", justifyContent: "center",
          backdropFilter: "blur(4px)", zIndex: 500,
        }}>
          <div style={{ color: "#6b7a99", fontSize: 13 }}>Computing route…</div>
        </div>
      )}
    </div>
  );
}