// @lovable.dev/vite-tanstack-config already includes the following — do NOT add them manually
// or the app will break with duplicate plugins:
//   - tanstackStart, viteReact, tailwindcss, tsConfigPaths, cloudflare (build-only),
//     componentTagger (dev-only), VITE_* env injection, @ path alias, React/TanStack dedupe,
//     error logger plugins, and sandbox detection (port/host/strictPort).
// You can pass additional config via defineConfig({ vite: { ... } }) if needed.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";

// Redirect TanStack Start's bundled server entry to src/server.ts (our SSR error wrapper).
// @cloudflare/vite-plugin builds from this — wrangler.jsonc main alone is insufficient.
export default defineConfig({
  tanstackStart: {
    server: { entry: "server" },
  },
  vite: {
    // @neo4j-nvl/* uses extensionless ESM imports that Node can't resolve during SSR.
    // Bundle them so Vite's resolver handles the paths (official boilerplate uses 1.1.0).
    ssr: {
      noExternal: [
        "@neo4j-nvl/react",
        "@neo4j-nvl/base",
        "@neo4j-nvl/interaction-handlers",
        "@neo4j-nvl/layout-workers",
      ],
    },
    optimizeDeps: {
      include: ["@neo4j-nvl/react", "@neo4j-nvl/base"],
    },
  },
});
