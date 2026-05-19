import { SidebarContent } from "./SidebarContent";

export function Sidebar() {
  return (
    <aside className="hidden md:flex w-72 shrink-0 h-svh flex-col border-r border-border/70 bg-sidebar">
      <SidebarContent />
    </aside>
  );
}
