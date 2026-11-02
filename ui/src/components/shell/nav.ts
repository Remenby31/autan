import { createContext, useContext } from "react";

/** Opens the navigation drawer of a narrow window (below 768 px); a no-op elsewhere. */
export const NavDrawer = createContext<() => void>(() => {});

export const useOpenNav = () => useContext(NavDrawer);
