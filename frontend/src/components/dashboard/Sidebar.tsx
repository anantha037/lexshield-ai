"use client";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { signOut, useSession } from "next-auth/react";
import { Scale, MessageSquare, FileSearch, LogOut, User } from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  {
    label: "Chat Intelligence",
    href: "/dashboard/chat",
    icon: MessageSquare,
    description: "Ask legal questions",
  },
  {
    label: "Document Deep-Dive",
    href: "/dashboard/analyze",
    icon: FileSearch,
    description: "Analyze contracts & FIRs",
  },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { data: session } = useSession();

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col h-screen sticky top-0">
      {/* Logo */}
      <div className="p-5 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="bg-amber-500 p-1.5 rounded-lg">
            <Scale className="h-5 w-5 text-slate-950" />
          </div>
          <div>
            <p className="font-bold text-white text-sm">LexShield AI</p>
            <p className="text-amber-500 text-xs">Legal Intelligence</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 p-3 space-y-1">
        {navItems.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all group",
                active
                  ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
                  : "text-slate-400 hover:bg-slate-800 hover:text-white"
              )}
            >
              <item.icon
                className={cn(
                  "h-4 w-4 shrink-0",
                  active
                    ? "text-amber-400"
                    : "text-slate-500 group-hover:text-white"
                )}
              />
              <div>
                <p className="text-sm font-medium">{item.label}</p>
                <p
                  className={cn(
                    "text-xs",
                    active ? "text-amber-500/70" : "text-slate-600"
                  )}
                >
                  {item.description}
                </p>
              </div>
            </Link>
          );
        })}
      </nav>

      {/* User + Logout */}
      <div className="p-3 border-t border-slate-800">
        <div className="flex items-center gap-3 px-3 py-2 mb-2">
          <div className="bg-slate-700 p-1.5 rounded-full">
            <User className="h-4 w-4 text-slate-300" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-white text-sm font-medium truncate">
              {session?.user?.name ?? "User"}
            </p>
            <p className="text-slate-500 text-xs truncate">
              {session?.user?.email ?? ""}
            </p>
          </div>
        </div>
        <button
          onClick={() => signOut({ callbackUrl: "/login" })}
          className="w-full flex items-center gap-2 px-3 py-2 text-slate-500 hover:text-red-400 hover:bg-red-950/30 rounded-lg transition-all text-sm"
        >
          <LogOut className="h-4 w-4" />
          Sign Out
        </button>
      </div>
    </aside>
  );
}