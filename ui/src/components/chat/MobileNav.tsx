import { useState } from "react";
import { Menu } from "lucide-react";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { SidebarContent } from "./SidebarContent";

export function MobileNav() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        aria-label="Open navigation"
        className="grid place-items-center size-9 rounded-lg hover:bg-secondary transition-colors text-foreground/80"
      >
        <Menu className="size-[18px]" />
      </button>

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent
          side="left"
          className="p-0 w-72 bg-sidebar border-r border-border/70 [&>button]:top-3.5 [&>button]:right-3.5"
        >
          <SheetTitle className="sr-only">Navigation</SheetTitle>
          <SidebarContent onClose={() => setOpen(false)} />
        </SheetContent>
      </Sheet>
    </>
  );
}
