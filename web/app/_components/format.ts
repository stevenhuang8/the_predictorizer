const UTC = { timeZone: "UTC" } as const;

export function utcDate(iso: string): Date {
  return new Date(`${iso}T00:00:00Z`);
}

export function monthYear(iso: string): string {
  return utcDate(iso).toLocaleDateString("en-US", {
    ...UTC,
    month: "short",
    year: "numeric",
  });
}

export function longDate(iso: string): string {
  return utcDate(iso).toLocaleDateString("en-US", {
    ...UTC,
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function pct(value: number, digits = 1): string {
  return `${value.toFixed(digits)}%`;
}
