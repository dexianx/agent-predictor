import { motion } from "framer-motion";
import type { PersonaDetail } from "@/lib/types";
import { getPersonaDisplayName } from "@/lib/persona-utils";
import type { ProductFields } from "@/components/chat/Composer";

type SampleProduct = ProductFields & { label: string; currency: "USD" | "NGN" };

// Products NOT in the knowledge graph — the system infers from similar items in the same category.
// Categories are real (exist in Neo4j) so retrieval still finds relevant review signal.
// Rich fields (brand/category/features) improve item-vector retrieval accuracy.
const SAMPLE_PRODUCTS: SampleProduct[] = [
  {
    label: "Anker 737 Power Bank 24000mAh",
    item_name: "Anker 737 Power Bank 24000mAh",
    item_description: "140W max output, 24000mAh capacity, charges laptops and phones simultaneously, LED display",
    item_brand: "Anker",
    item_category: "Electronics / Power Banks",
    item_features: "140W max output\n24000mAh capacity\nCharges 3 devices simultaneously\nLED battery display\nFoldable prongs",
    price_ngn: 89.99,
    currency: "USD",
  },
  {
    label: "Logitech MX Keys S Keyboard",
    item_name: "Logitech MX Keys S Wireless Keyboard",
    item_description: "Full-size backlit wireless keyboard, multi-device Bluetooth, USB-C charging",
    item_brand: "Logitech",
    item_category: "Computers / Keyboards",
    item_features: "Multi-device Bluetooth (up to 3 devices)\nBacklit spherically-dished keys\nUSB-C charging\nWorks on Windows, Mac, Linux\n10-day battery life",
    price_ngn: 109.99,
    currency: "USD",
  },
  {
    label: "Hydro Flask 32 oz Water Bottle",
    item_name: "Hydro Flask 32oz Wide Mouth Water Bottle",
    item_description: "Insulated stainless steel bottle, flex cap, keeps cold 24hrs and hot 12hrs",
    item_brand: "Hydro Flask",
    item_category: "Sports & Outdoors / Water Bottles",
    price_ngn: 49.95,
    currency: "USD",
  },
  {
    label: "DJI Osmo Mobile 6 Gimbal",
    item_name: "DJI Osmo Mobile 6 Smartphone Gimbal",
    item_description: "3-axis stabilizer with built-in extension rod, ActiveTrack 6.0, foldable design",
    item_brand: "DJI",
    item_category: "Camera & Photo / Gimbals",
    item_features: "3-axis stabilization\nBuilt-in extension rod\nActiveTrack 6.0\nFoldable design\nShotGuides creative templates",
    price_ngn: 159.00,
    currency: "USD",
  },
  {
    label: "Spigen Case for Galaxy S25",
    item_name: "Spigen Tough Armor Samsung Galaxy S25 Case",
    item_description: "Dual-layer military-grade protection, kickstand, compatible with MagSafe",
    item_brand: "Spigen",
    item_category: "Cell Phones & Accessories / Cases",
    price_ngn: 19.99,
    currency: "USD",
  },
  {
    label: "Cosori 5.8QT Air Fryer",
    item_name: "Cosori Air Fryer 5.8QT",
    item_description: "5.8-quart square basket, 9 one-touch cooking functions, dishwasher-safe parts",
    item_brand: "Cosori",
    item_category: "Kitchen Appliances / Air Fryers",
    item_features: "5.8-quart square non-stick basket\n9 one-touch cooking presets\nDishwasher-safe basket and crisper plate\n1700W heating element\nAuto shutoff safety feature",
    item_details: "Wattage: 1700W · Capacity: 5.8 qt · Temperature range: 170–400°F · Dimensions: 11.8 × 11.8 × 12.6 in",
    price_ngn: 99.99,
    currency: "USD",
  },
];

const EASE = [0.22, 1, 0.36, 1] as const;

export function EmptyState({
  persona,
  onPick,
}: {
  persona: PersonaDetail | null;
  onPick: (fields: ProductFields) => void;
}) {
  const displayName = persona ? getPersonaDisplayName(persona) : "this persona";
  return (
    <div className="max-w-3xl mx-auto px-4 md:px-6 pt-12 md:pt-24 pb-10">
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, ease: EASE }}
      >
        <h1 className="font-serif text-[clamp(28px,7vw,40px)] leading-[1.1] tracking-tight text-foreground">
          Good to see you<span className="text-primary">.</span>
        </h1>
        <p className="mt-3 text-ink-soft text-[14.5px] md:text-[15.5px] leading-relaxed max-w-xl">
          Pick a product below and Kora will generate a review in{" "}
          <span className="text-foreground font-medium">{displayName}</span>'s authentic voice.
        </p>
      </motion.div>

      <motion.div
        initial="hidden"
        animate="show"
        variants={{
          hidden: {},
          show: { transition: { staggerChildren: 0.04, delayChildren: 0.2 } },
        }}
        className="mt-8 md:mt-10 flex flex-col"
      >
        {SAMPLE_PRODUCTS.map((product) => (
          <motion.button
            key={product.item_name}
            variants={{
              hidden: { opacity: 0, y: 4 },
              show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: EASE } },
            }}
            onClick={() => onPick(product)}
            className="group text-left py-3 border-t border-border/60 last:border-b hover:text-foreground active:text-foreground transition-colors flex items-center justify-between gap-4 min-h-[52px]"
          >
            <span className="flex flex-col gap-0.5 min-w-0">
              <span className="text-[14.5px] md:text-[15px] text-ink-soft group-hover:text-foreground transition-colors truncate">
                {product.label}
              </span>
              <span className="text-[12px] text-muted-foreground/60 truncate">
                {product.currency === "USD" ? "$" : "₦"}{product.price_ngn?.toLocaleString()} · {product.item_description}
              </span>
            </span>
            <span
              aria-hidden
              className="shrink-0 text-muted-foreground/40 group-hover:text-foreground/60 group-hover:translate-x-0.5 transition-all"
            >
              ↗
            </span>
          </motion.button>
        ))}
      </motion.div>
    </div>
  );
}
