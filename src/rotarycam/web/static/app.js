(() => {
  "use strict";

  const STEP_SELECTOR = "[data-step-select]";
  const PANEL_SELECTOR = "[data-step-panel]";
  const SCENE_EVENT = "rotarycam:scene-changed";
  let lastCompletedRevision = null;
  let viewerModulePromise = null;

  async function ensureViewer() {
    const scene = document.getElementById("scene-viewer");
    if (!scene) return;
    scene.dataset.viewerState = "loading";
    try {
      viewerModulePromise ||= import("/static/viewer.js?v=1");
      const viewerModule = await viewerModulePromise;
      viewerModule.initializeViewer();
      scene.dataset.viewerState = "ready";
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to start the 3D preview.";
      scene.dataset.viewerState = "error";
      scene.dataset.viewerError = message;
      console.error("RotaryCAM 3D preview failed:", error);
    }
  }

  function announce(message) {
    const region = document.getElementById("status-region");
    if (!region || !message) return;
    region.textContent = message;
    region.classList.add("is-visible");
    window.clearTimeout(announce.timeout);
    announce.timeout = window.setTimeout(() => region.classList.remove("is-visible"), 3500);
  }

  function availableStepButton(step) {
    return Array.from(document.querySelectorAll(STEP_SELECTOR)).find(
      (button) => button.dataset.stepSelect === step && button.getAttribute("aria-disabled") !== "true",
    );
  }

  function selectStep(step, { focusPanel = false } = {}) {
    const selected = availableStepButton(step);
    if (!selected) {
      const blocked = Array.from(document.querySelectorAll(STEP_SELECTOR)).find(
        (button) => button.dataset.stepSelect === step,
      );
      if (blocked) announce(blocked.title || "Complete the previous step first.");
      return;
    }

    document.querySelectorAll(STEP_SELECTOR).forEach((button) => {
      const active = button.dataset.stepSelect === step;
      button.setAttribute("aria-selected", String(active));
      if (button.closest(".workflow-list")) button.tabIndex = active ? 0 : -1;
    });
    document.querySelectorAll(PANEL_SELECTOR).forEach((panel) => {
      const active = panel.dataset.stepPanel === step;
      panel.hidden = !active;
      panel.setAttribute("aria-hidden", String(!active));
    });

    const shell = document.getElementById("workspace-shell");
    if (shell) shell.dataset.activeStep = step;
    if (focusPanel) document.querySelector(`[data-step-panel="${CSS.escape(step)}"]`)?.focus();
  }

  function currentStep() {
    return (
      document.querySelector(`${STEP_SELECTOR}[aria-selected="true"]`)?.dataset.stepSelect ||
      document.getElementById("workspace-shell")?.dataset.activeStep ||
      "model"
    );
  }

  function initializeWorkspace(root = document) {
    const preferred = root.querySelector?.(`${STEP_SELECTOR}[aria-selected="true"]`)?.dataset.stepSelect;
    const shell = document.getElementById("workspace-shell");
    if (shell) selectStep(preferred || shell.dataset.activeStep || "model");
    updateConditionalFields(root);
    detectJobCompletion(root);
    void ensureViewer();
  }

  function updateConditionalFields(root = document) {
    const stockKind = root.querySelector?.('input[name="type"][value="cylinder"]:checked')
      ? "cylinder"
      : root.querySelector?.('input[name="type"][value="rectangle"]:checked')
        ? "rectangle"
        : null;
    root.querySelectorAll?.("[data-stock-shape]").forEach((field) => {
      const visible = field.dataset.stockShape === stockKind;
      field.hidden = !visible;
      field.querySelectorAll("input").forEach((input) => {
        input.disabled = !visible;
        input.required = visible;
      });
    });

    const supportForm = root.querySelector?.("#support-form") || document.getElementById("support-form");
    if (!supportForm) return;
    const supportKind = supportForm.querySelector('input[name="type"]:checked')?.value;
    supportForm.querySelectorAll("[data-support-shape]").forEach((field) => {
      const visible = field.dataset.supportShape === supportKind;
      field.hidden = !visible;
      field.querySelectorAll("input").forEach((input) => {
        input.disabled = !visible;
        input.required = visible;
      });
    });
  }

  function notifyScene(reason, revision = null) {
    const scene = document.getElementById("scene-viewer");
    if (!scene) return;
    scene.dispatchEvent(
      new CustomEvent(SCENE_EVENT, { bubbles: true, detail: { reason, revision } }),
    );
  }

  function detectJobCompletion(root = document) {
    const panel = root.querySelector?.("#job-panel") || document.getElementById("job-panel");
    if (!panel || panel.dataset.jobState !== "completed") return;
    const revision = panel.dataset.jobRevision || "unknown";
    if (lastCompletedRevision === revision) return;
    lastCompletedRevision = revision;
    announce("Toolpaths generated. Review the scene and validation results.");
    notifyScene("job-completed", revision);
  }

  function applyPickedSupport(detail) {
    const form = document.getElementById("support-form");
    if (!form || !detail) return;
    const x = Number(detail.x);
    const y = Number(detail.y);
    const z = Number(detail.z);
    const suppliedAngle = Number(detail.angleDeg);
    const angle = Number.isFinite(suppliedAngle)
      ? suppliedAngle
      : ((Math.atan2(z, y) * 180) / Math.PI + 360) % 360;

    if (Number.isFinite(x)) form.elements.namedItem("x").value = x.toFixed(3);
    if (Number.isFinite(angle)) form.elements.namedItem("angle_deg").value = angle.toFixed(3);
    if (Number.isFinite(y)) form.elements.namedItem("pick_y").value = y.toFixed(6);
    if (Number.isFinite(z)) form.elements.namedItem("pick_z").value = z.toFixed(6);
    selectStep("supports", { focusPanel: true });
    announce("Support position captured from the 3D view.");
  }

  document.addEventListener("click", (event) => {
    const stepButton = event.target.closest(STEP_SELECTOR);
    if (stepButton) {
      event.preventDefault();
      selectStep(stepButton.dataset.stepSelect, { focusPanel: !stepButton.closest(".workflow-list") });
      return;
    }

    const axisButton = event.target.closest("[data-view-axis]");
    if (axisButton) {
      document.querySelectorAll("[data-view-axis]").forEach((button) => {
        button.setAttribute("aria-pressed", String(button === axisButton));
      });
    }
  });

  document.addEventListener("keydown", (event) => {
    const active = event.target.closest?.(".workflow-list [data-step-select]");
    if (!active || !["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    const steps = Array.from(document.querySelectorAll(".workflow-list [data-step-select]"));
    let index = steps.indexOf(active);
    if (event.key === "Home") index = 0;
    else if (event.key === "End") index = steps.length - 1;
    else index = (index + (event.key === "ArrowDown" ? 1 : -1) + steps.length) % steps.length;
    event.preventDefault();
    steps[index].focus();
  });

  document.addEventListener("change", (event) => {
    if (event.target.matches('input[name="type"]')) updateConditionalFields(document);
  });

  document.addEventListener("rotarycam:support-picked", (event) => applyPickedSupport(event.detail));
  document.addEventListener(SCENE_EVENT, () => void ensureViewer());

  document.addEventListener("htmx:configRequest", (event) => {
    const token = document.querySelector('meta[name="csrf-token"]')?.content;
    if (token) event.detail.headers["X-CSRF-Token"] = token;
  });

  document.addEventListener("htmx:beforeRequest", (event) => {
    const trigger = event.detail.elt || event.target;
    trigger?.classList.add("is-busy");
    trigger?.setAttribute("aria-busy", "true");
  });

  document.addEventListener("htmx:afterSwap", (event) => {
    const preservedStep = currentStep();
    initializeWorkspace(document);
    if (availableStepButton(preservedStep)) selectStep(preservedStep);
    detectJobCompletion(document);
    const path = event.detail.pathInfo?.requestPath || event.detail.path || "";
    if (/\/(mesh|stock|tools|settings|machine|supports)(\/|$)/.test(path)) {
      notifyScene("workspace-updated");
    }
  });

  document.addEventListener("htmx:afterRequest", (event) => {
    const trigger = event.detail.elt || event.target;
    trigger?.classList.remove("is-busy");
    trigger?.removeAttribute("aria-busy");
    if (!event.detail.successful) {
      announce("That change could not be saved. Check the highlighted values and try again.");
      return;
    }
  });

  document.addEventListener("DOMContentLoaded", () => {
    document.documentElement.dataset.rotarycamApp = "ready";
    initializeWorkspace(document);
    notifyScene("initial-load");
  });
})();
