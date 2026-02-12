import { atom } from "jotai";
import { tourSteps, type TourStep } from "@/data/tourSteps";

const TOUR_STORAGE_KEY = "layout-tour";

// ── Core state atoms ──
export const tourActiveAtom = atom(false);
export const tourStepIndexAtom = atom(0);
export const tourHasSeenAtom = atom(false);

// ── Derived atoms ──
export const currentTourStepAtom = atom<TourStep | null>((get) =>
  get(tourActiveAtom) ? tourSteps[get(tourStepIndexAtom)] ?? null : null,
);

export const tourTotalStepsAtom = atom(tourSteps.length);

export const tourIsFirstStepAtom = atom((get) => get(tourStepIndexAtom) === 0);

export const tourIsLastStepAtom = atom(
  (get) => get(tourStepIndexAtom) === tourSteps.length - 1,
);

export const tourProgressAtom = atom(
  (get) => ((get(tourStepIndexAtom) + 1) / tourSteps.length) * 100,
);

// ── Action atoms ──
export const startTourAtom = atom(null, (_get, set) => {
  set(tourStepIndexAtom, 0);
  set(tourActiveAtom, true);
});

export const completeTourAtom = atom(null, (_get, set) => {
  set(tourActiveAtom, false);
  set(tourHasSeenAtom, true);
  persistTourState({ completed: true });
});

export const nextTourStepAtom = atom(null, (get, set) => {
  const idx = get(tourStepIndexAtom);
  if (idx < tourSteps.length - 1) {
    set(tourStepIndexAtom, idx + 1);
  } else {
    set(completeTourAtom);
  }
});

export const prevTourStepAtom = atom(null, (get, set) => {
  const idx = get(tourStepIndexAtom);
  if (idx > 0) {
    set(tourStepIndexAtom, idx - 1);
  }
});

export const goToTourStepAtom = atom(null, (_get, set, index: number) => {
  if (index >= 0 && index < tourSteps.length) {
    set(tourStepIndexAtom, index);
  }
});

export const skipTourAtom = atom(null, (_get, set) => {
  set(tourActiveAtom, false);
  set(tourHasSeenAtom, true);
  persistTourState({ skipped: true });
});

export const resetTourAtom = atom(null, (_get, set) => {
  set(tourStepIndexAtom, 0);
  set(tourHasSeenAtom, false);
  set(tourActiveAtom, false);
  clearTourStorage();
});

// ── Persistence helpers ──
function persistTourState(options: { completed?: boolean; skipped?: boolean } = {}) {
  try {
    const state = {
      hasSeenTour: true,
      completed: options.completed ?? false,
      skipped: options.skipped ?? false,
      timestamp: Date.now(),
    };
    localStorage.setItem(TOUR_STORAGE_KEY, JSON.stringify(state));
  } catch {
    // localStorage unavailable (SSR or private browsing)
  }
}

export function loadTourState(): { hasSeenTour: boolean } {
  try {
    const saved = localStorage.getItem(TOUR_STORAGE_KEY);
    if (saved) {
      const state = JSON.parse(saved);
      return {
        hasSeenTour: state.hasSeenTour || state.completed || state.skipped || false,
      };
    }
  } catch {
    // localStorage unavailable
  }
  return { hasSeenTour: false };
}

function clearTourStorage() {
  try {
    localStorage.removeItem(TOUR_STORAGE_KEY);
  } catch {
    // localStorage unavailable
  }
}
