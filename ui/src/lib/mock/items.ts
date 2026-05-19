export type Item = {
  id: string;
  dataset: "yelp" | "amazon" | "goodreads";
  title: string;
  subtitle: string; // author / brand / cuisine
  tags: string[];
  blurb: string;
  avgRating: number;
  reviews: number;
  price?: string;
  emoji: string;
};

export const yelp: Item[] = [
  { id: "y1", dataset: "yelp", title: "Iya Moria Buka", subtitle: "Yoruba · Yaba", tags: ["jollof", "amala", "outdoor"], blurb: "Smoky party-rice institution tucked behind Sabo market.", avgRating: 4.6, reviews: 812, price: "₦", emoji: "🍛" },
  { id: "y2", dataset: "yelp", title: "Nok by Alara", subtitle: "Pan-African · Victoria Island", tags: ["fine dining", "date night", "design-y"], blurb: "Architect-designed space serving a refined West African tasting menu.", avgRating: 4.4, reviews: 401, price: "₦₦₦₦", emoji: "🍷" },
  { id: "y3", dataset: "yelp", title: "Bature Brewery", subtitle: "Gastropub · Lekki", tags: ["craft beer", "rooftop", "live music"], blurb: "Lagos's first craft brewery with rotating taps and a chill rooftop.", avgRating: 4.2, reviews: 623, price: "₦₦₦", emoji: "🍺" },
  { id: "y4", dataset: "yelp", title: "Calabash Suya Spot", subtitle: "Hausa · Wuse II", tags: ["suya", "late night", "casual"], blurb: "Charcoal-grilled suya wrapped in newsprint, exactly as it should be.", avgRating: 4.5, reviews: 287, price: "₦", emoji: "🍢" },
  { id: "y5", dataset: "yelp", title: "Cafe Neo Akin Adesola", subtitle: "Cafe · Victoria Island", tags: ["coffee", "wifi", "work-friendly"], blurb: "Reliable single-origin pour-overs and quiet enough to take a meeting.", avgRating: 4.0, reviews: 911, price: "₦₦", emoji: "☕" },
  { id: "y6", dataset: "yelp", title: "Cactus Restaurant", subtitle: "Continental · Ikoyi", tags: ["breakfast", "pastries", "garden"], blurb: "Garden-side breakfasts. The pancakes are the move.", avgRating: 4.3, reviews: 540, price: "₦₦", emoji: "🥞" },
  { id: "y7", dataset: "yelp", title: "RSVP Lagos", subtitle: "Modern Nigerian · VI", tags: ["small plates", "cocktails", "loud"], blurb: "Plantain gnocchi, jollof arancini — playful, sometimes overcooked.", avgRating: 4.1, reviews: 333, price: "₦₦₦", emoji: "🍸" },
  { id: "y8", dataset: "yelp", title: "Mama Put Surulere", subtitle: "Local · Surulere", tags: ["soup", "ewedu", "value"], blurb: "Unbeatable ewedu and gbegiri at unbeatable prices.", avgRating: 4.7, reviews: 1204, price: "₦", emoji: "🥣" },
];

export const amazon: Item[] = [
  { id: "a1", dataset: "amazon", title: "Anker Soundcore Liberty 4", subtitle: "Anker · Audio", tags: ["wireless", "ANC", "value"], blurb: "Surprisingly capable ANC at a friendly price.", avgRating: 4.4, reviews: 21430, price: "$99", emoji: "🎧" },
  { id: "a2", dataset: "amazon", title: "Kindle Paperwhite (12th gen)", subtitle: "Amazon · E-readers", tags: ["e-ink", "waterproof", "long battery"], blurb: "Brighter screen, faster page turns, the e-reader to beat.", avgRating: 4.7, reviews: 88210, price: "$159", emoji: "📖" },
  { id: "a3", dataset: "amazon", title: "Logitech MX Master 3S", subtitle: "Logitech · Peripherals", tags: ["productivity", "ergonomic", "wireless"], blurb: "Cross-device wizardry; the scroll wheel is a tiny religion.", avgRating: 4.6, reviews: 15302, price: "$99", emoji: "🖱️" },
  { id: "a4", dataset: "amazon", title: "Oraimo Power Bank 30000mAh", subtitle: "Oraimo · Mobile", tags: ["high-capacity", "fast-charge", "Africa-popular"], blurb: "Heavy, but it'll see you through a NEPA week.", avgRating: 4.3, reviews: 4201, price: "$45", emoji: "🔋" },
  { id: "a5", dataset: "amazon", title: "Nintendo Switch OLED", subtitle: "Nintendo · Gaming", tags: ["handheld", "OLED", "family"], blurb: "Same Switch, much nicer screen.", avgRating: 4.8, reviews: 67100, price: "$349", emoji: "🎮" },
  { id: "a6", dataset: "amazon", title: "Hario V60 Ceramic Dripper", subtitle: "Hario · Coffee", tags: ["pour-over", "ritual", "compact"], blurb: "The pour-over standard. You will overthink your bloom.", avgRating: 4.7, reviews: 9100, price: "$25", emoji: "☕" },
];

export const goodreads: Item[] = [
  { id: "g1", dataset: "goodreads", title: "Half of a Yellow Sun", subtitle: "Chimamanda Ngozi Adichie", tags: ["historical", "Biafra", "literary"], blurb: "A devastating, intimate epic of the Biafran war.", avgRating: 4.4, reviews: 251000, emoji: "📕" },
  { id: "g2", dataset: "goodreads", title: "Things Fall Apart", subtitle: "Chinua Achebe", tags: ["classic", "colonialism", "Igbo"], blurb: "The cornerstone of modern African literature.", avgRating: 3.8, reviews: 380000, emoji: "📗" },
  { id: "g3", dataset: "goodreads", title: "The Fishermen", subtitle: "Chigozie Obioma", tags: ["myth", "brothers", "tragedy"], blurb: "A myth-soaked family tragedy by a riverbank in Akure.", avgRating: 3.9, reviews: 32100, emoji: "📘" },
  { id: "g4", dataset: "goodreads", title: "Stay With Me", subtitle: "Ayọ̀bámi Adébáyọ̀", tags: ["marriage", "Yoruba", "grief"], blurb: "A marriage cracking quietly under the weight of expectation.", avgRating: 4.1, reviews: 41200, emoji: "📙" },
  { id: "g5", dataset: "goodreads", title: "My Sister, the Serial Killer", subtitle: "Oyinkan Braithwaite", tags: ["dark comedy", "Lagos", "fast"], blurb: "Sisterhood with a body count. Dispatched in one sitting.", avgRating: 3.7, reviews: 210400, emoji: "📓" },
  { id: "g6", dataset: "goodreads", title: "Freshwater", subtitle: "Akwaeke Emezi", tags: ["spirituality", "identity", "experimental"], blurb: "Ogbanje cosmology meets debut autofiction.", avgRating: 3.9, reviews: 24600, emoji: "📔" },
  { id: "g7", dataset: "goodreads", title: "Beasts of No Nation", subtitle: "Uzodinma Iweala", tags: ["war", "child soldier", "raw"], blurb: "A child soldier's voice you cannot put down or shake off.", avgRating: 4.0, reviews: 19800, emoji: "📒" },
];

export const allItems: Item[] = [...yelp, ...amazon, ...goodreads];

export const itemsByDataset = (d: Item["dataset"]) =>
  d === "yelp" ? yelp : d === "amazon" ? amazon : goodreads;

export const itemById = (id: string) => allItems.find((i) => i.id === id);
