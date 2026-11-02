import { createContext, useContext } from "react";

/** The navigation drawer of a narrow window (below 768 px): whether it is open, and how to open
 * it; its element has the id `nav-drawer`. */
export const NavDrawer = createContext<{ open: boolean; openNav: () => void }>({
  open: false,
  openNav: () => {},
});

export const useNavDrawer = () => useContext(NavDrawer);
