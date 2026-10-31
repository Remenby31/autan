/** A diagram from `docs/assets/diagrams/`, inlined by the build so it follows the page theme. */
export function Diagram({ svg, alt }: { svg: string; alt: string }) {
  return (
    <span
      role="img"
      aria-label={alt}
      className="my-6 block [&>svg]:h-auto [&>svg]:w-full"
      // biome-ignore lint/security/noDangerouslySetInnerHtml: the build reads the SVG from docs/assets/diagrams in this repository.
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
