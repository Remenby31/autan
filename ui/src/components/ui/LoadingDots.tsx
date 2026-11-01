/** Three pulsing dots for work in progress inside text or a button. */
export function LoadingDots({ label = "Loading" }: { label?: string }) {
  return (
    <span role="status" aria-label={label} className="inline-flex items-center gap-[3px]">
      {[0, 1, 2].map((k) => (
        <span
          key={k}
          className="size-1 rounded-full bg-current"
          style={{
            animation: "dots 1.4s ease-in-out infinite",
            animationDelay: `${k * 200}ms`,
          }}
        />
      ))}
    </span>
  );
}
