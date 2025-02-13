import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk

# ====================================
# Theme & Style Configuration
# ====================================
# NOTE: Tweaked the colors and added more consistency.
PRIMARY_BG   = "#003366"    # Dark blue background for primary sections
SECONDARY_BG = "#00509E"    # Lighter blue for secondary panels
ACCENT_BG    = "#E6F2FF"    # Very light blue accent (for tooltips and highlights)
TEXT_COLOR   = "#ffffff"    # White text for dark backgrounds
BUTTON_BG    = "#ffffff"    # White button background for contrast
BUTTON_FG    = PRIMARY_BG   # Dark blue text for buttons
LABEL_COLOR  = "#c0c0c0"    # Light grey for labels

HEADER_FONT  = ("Helvetica", 18, "bold")
LABEL_FONT   = ("Helvetica", 10)
ENTRY_FONT   = ("Helvetica", 12)

# Padding for widgets
PAD_X = 10
PAD_Y = 5

# ====================================
# Tooltip Class
# ====================================
class ToolTip:
    """Create a tooltip for a given widget."""
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        if self.tipwindow or not self.text:
            return
        # NOTE: Adjusted tooltip offset for a cleaner look.
        x, y, _, _ = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 20
        y = y + self.widget.winfo_rooty() + 10
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background=ACCENT_BG, relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"), wraplength=200)
        label.pack(ipadx=4, ipady=2)

    def hide(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
        self.tipwindow = None

# ====================================
# Monte Carlo Option Pricer (Multiple Models)
# ====================================
class MonteCarloOptionPricer:
    """
    Option pricer using Monte Carlo simulation with different models.
    
    Models supported:
      - "GBM": Geometric Brownian Motion.
      - "Jump Diffusion": Merton jump diffusion model.
    
    For Jump Diffusion, additional parameters (jump intensity, mean, volatility)
    are used.
    """
    def __init__(self, S0, K, T, r, sigma, N=100000, num_paths=5, steps=252,
                 model="GBM", jump_intensity=0, jump_mean=0, jump_vol=0):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.num_paths = num_paths
        self.steps = steps
        self.model = model
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_vol = jump_vol

    def simulate_paths(self):
        dt = self.T / self.steps
        paths = np.zeros((self.num_paths, self.steps + 1))
        paths[:, 0] = self.S0
        if self.model == "GBM":
            for t in range(1, self.steps + 1):
                Z = np.random.standard_normal(self.num_paths)
                paths[:, t] = paths[:, t - 1] * np.exp(
                    (self.r - 0.5 * self.sigma**2) * dt +
                    self.sigma * np.sqrt(dt) * Z
                )
        elif self.model == "Jump Diffusion":
            for t in range(1, self.steps + 1):
                Z = np.random.standard_normal(self.num_paths)
                for i in range(self.num_paths):
                    jump_count = np.random.poisson(self.jump_intensity * dt)
                    jump_sum = np.sum(np.random.normal(self.jump_mean, self.jump_vol, jump_count)) if jump_count > 0 else 0
                    drift_adjustment = (self.r - self.jump_intensity * 
                                        (np.exp(self.jump_mean + 0.5 * self.jump_vol**2) - 1)
                                        - 0.5 * self.sigma**2) * dt
                    paths[i, t] = paths[i, t-1] * np.exp(
                        drift_adjustment + self.sigma * np.sqrt(dt) * Z[i] + jump_sum
                    )
        return paths

    def price_option(self, option_type="Call"):
        # Use "Put" if option_type.lower() == "put", otherwise "Call"
        if option_type.lower() == "put":
            payoff_func = lambda ST: np.maximum(self.K - ST, 0)
        else:
            payoff_func = lambda ST: np.maximum(ST - self.K, 0)
        if self.model == "GBM":
            Z = np.random.standard_normal(self.N)
            ST = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * self.T +
                                  self.sigma * np.sqrt(self.T) * Z)
        elif self.model == "Jump Diffusion":
            Z = np.random.standard_normal(self.N)
            jump_counts = np.random.poisson(self.jump_intensity * self.T, self.N)
            jumps = np.array([np.sum(np.random.normal(self.jump_mean, self.jump_vol, int(jc))) if jc > 0 else 0 
                              for jc in jump_counts])
            ST = self.S0 * np.exp(
                (self.r - self.jump_intensity * (np.exp(self.jump_mean + 0.5*self.jump_vol**2) - 1) - 0.5*self.sigma**2)
                * self.T +
                self.sigma * np.sqrt(self.T) * Z + jumps
            )
        payoffs = payoff_func(ST)
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.N)
        confidence_interval = (option_price - 1.96 * std_error,
                               option_price + 1.96 * std_error)
        return option_price, confidence_interval, ST

# ====================================
# Global Variables for Plot Canvases
# ====================================
sim_canvas = None
hist_canvas = None
heatmap_canvas = None

# ====================================
# UI Functions
# ====================================
def get_input_values():
    """
    Validates and retrieves input values from the UI.
    Returns a tuple with all parameters or None if invalid.
    """
    try:
        S0 = float(entry_S0.get())
        K = float(entry_K.get())
        T = float(entry_T.get())
        r = float(entry_r.get()) / 100  # convert percentage to decimal
        sigma = float(entry_sigma.get()) / 100
        N = int(spinbox_N.get())
        option_type = combobox_option.get()
        model = combobox_model.get()
        num_paths = int(spinbox_paths.get())
        steps = int(spinbox_steps.get())
        if model == "Jump Diffusion":
            jump_intensity = float(entry_jump_intensity.get())
            jump_mean = float(entry_jump_mean.get())
            jump_vol = float(entry_jump_vol.get())
        else:
            jump_intensity, jump_mean, jump_vol = 0, 0, 0
        return (S0, K, T, r, sigma, N, option_type, model, num_paths, steps,
                jump_intensity, jump_mean, jump_vol)
    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid input detected. Please check your entries.\nError: {e}")
        return None

def run_simulation():
    """
    Runs the simulation and embeds the stock price paths plot into the Simulation tab.
    """
    inputs = get_input_values()
    if not inputs:
        return
    try:
        (S0, K, T, r, sigma, N, option_type, model, num_paths, steps,
         jump_intensity, jump_mean, jump_vol) = inputs
        pricer = MonteCarloOptionPricer(S0, K, T, r, sigma, N, num_paths, steps,
                                        model, jump_intensity, jump_mean, jump_vol)
        paths = pricer.simulate_paths()
    
        # Create a larger figure with white background and grid lines for better readability.
        fig = Figure(figsize=(9, 6), facecolor="white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        for path in paths:
            ax.plot(path, color=PRIMARY_BG, alpha=0.8, lw=1.5)
        ax.set_title("Simulated Stock Price Paths", color=PRIMARY_BG, fontsize=14)
        ax.set_xlabel("Time Steps", color=PRIMARY_BG, fontsize=12)
        ax.set_ylabel("Stock Price", color=PRIMARY_BG, fontsize=12)
        ax.tick_params(axis='x', colors=PRIMARY_BG)
        ax.tick_params(axis='y', colors=PRIMARY_BG)
        ax.grid(True, linestyle="--", alpha=0.5)
    
        global sim_canvas
        if sim_canvas is not None:
            sim_canvas.get_tk_widget().destroy()
        sim_canvas = FigureCanvasTkAgg(fig, master=sim_frame)
        sim_canvas.draw()
        sim_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=PAD_Y)
        status_var.set("Simulation complete.")
    except Exception as e:
        messagebox.showerror("Simulation Error", f"An error occurred during simulation.\n{e}")

def show_option_price():
    """
    Computes the option price, updates the Option Pricing tab, and displays a histogram.
    """
    inputs = get_input_values()
    if not inputs:
        return
    try:
        (S0, K, T, r, sigma, N, option_type, model, num_paths, steps,
         jump_intensity, jump_mean, jump_vol) = inputs
        pricer = MonteCarloOptionPricer(S0, K, T, r, sigma, N, num_paths, steps,
                                        model, jump_intensity, jump_mean, jump_vol)
        price, conf_interval, ST = pricer.price_option(option_type)
        result_text = (f"Option Type: {option_type}\n"
                       f"Model: {model}\n"
                       f"Estimated Price: ${price:.4f}\n"
                       f"95% Confidence Interval: (${conf_interval[0]:.4f}, ${conf_interval[1]:.4f})")
        lbl_result.config(text=result_text)
        update_histogram(ST)
        status_var.set("Option pricing complete.")
    except Exception as e:
        messagebox.showerror("Pricing Error", f"An error occurred during option pricing.\n{e}")

def update_histogram(ST):
    """
    Plots a histogram of the simulated final stock prices.
    """
    try:
        fig = Figure(figsize=(9, 6), facecolor="white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        ax.hist(ST, bins=30, color=PRIMARY_BG, edgecolor=PRIMARY_BG, alpha=0.8)
        ax.set_title("Distribution of Final Stock Prices", color=PRIMARY_BG, fontsize=14)
        ax.set_xlabel("Final Stock Price", color=PRIMARY_BG, fontsize=12)
        ax.set_ylabel("Frequency", color=PRIMARY_BG, fontsize=12)
        ax.tick_params(axis='x', colors=PRIMARY_BG)
        ax.tick_params(axis='y', colors=PRIMARY_BG)
        ax.grid(True, linestyle="--", alpha=0.5)
    
        global hist_canvas
        if hist_canvas is not None:
            hist_canvas.get_tk_widget().destroy()
        hist_canvas = FigureCanvasTkAgg(fig, master=hist_frame)
        hist_canvas.draw()
        hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=PAD_Y)
    except Exception as e:
        messagebox.showerror("Histogram Error", f"An error occurred while updating the histogram.\n{e}")

def run_heatmap():
    """
    Simulates paths and creates a heatmap showing the density of stock prices over time.
    """
    inputs = get_input_values()
    if not inputs:
        return
    try:
        (S0, K, T, r, sigma, N, option_type, model, _, steps,
         jump_intensity, jump_mean, jump_vol) = inputs
        heatmap_paths = 500  # More paths for a smoother heatmap
        pricer = MonteCarloOptionPricer(S0, K, T, r, sigma, N,
                                        heatmap_paths, steps,
                                        model, jump_intensity, jump_mean, jump_vol)
        paths = pricer.simulate_paths()  # shape: (heatmap_paths, steps+1)
    
        price_min = np.min(paths)
        price_max = np.max(paths)
        bins = 50
        bin_edges = np.linspace(price_min, price_max, bins+1)
        heatmap_matrix = np.zeros((bins, steps+1))
        for t in range(steps+1):
            hist, _ = np.histogram(paths[:, t], bins=bin_edges)
            heatmap_matrix[:, t] = hist
    
        fig = Figure(figsize=(9, 6), facecolor="white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        im = ax.imshow(heatmap_matrix, aspect='auto', origin='lower',
                       cmap='viridis', extent=[0, steps, price_min, price_max])
        ax.set_title("Heatmap of Simulated Prices", color=PRIMARY_BG, fontsize=14)
        ax.set_xlabel("Time Steps", color=PRIMARY_BG, fontsize=12)
        ax.set_ylabel("Stock Price", color=PRIMARY_BG, fontsize=12)
        ax.tick_params(axis='x', colors=PRIMARY_BG)
        ax.tick_params(axis='y', colors=PRIMARY_BG)
        fig.colorbar(im, ax=ax)
    
        global heatmap_canvas
        if heatmap_canvas is not None:
            heatmap_canvas.get_tk_widget().destroy()
        heatmap_canvas = FigureCanvasTkAgg(fig, master=heatmap_frame)
        heatmap_canvas.draw()
        heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=PAD_Y)
        status_var.set("Heatmap updated.")
    except Exception as e:
        messagebox.showerror("Heatmap Error", f"An error occurred while updating the heatmap.\n{e}")

def reset_inputs():
    """
    Resets all input fields to default values relevant to finance.
    Defaults: S0 = 100, K = 100, T = 1, r = 2.5%, σ = 20%, 100,000 simulations,
    Call option, GBM model, 10 display paths, and 252 steps.
    """
    entry_S0.delete(0, tk.END);      entry_S0.insert(0, "100")
    entry_K.delete(0, tk.END);       entry_K.insert(0, "100")
    entry_T.delete(0, tk.END);       entry_T.insert(0, "1")
    entry_r.delete(0, tk.END);       entry_r.insert(0, "2.5")
    entry_sigma.delete(0, tk.END);   entry_sigma.insert(0, "20")
    spinbox_N.delete(0, tk.END);     spinbox_N.insert(0, "100000")
    combobox_option.set("Call")
    combobox_model.set("GBM")
    spinbox_paths.delete(0, tk.END); spinbox_paths.insert(0, "10")
    spinbox_steps.delete(0, tk.END); spinbox_steps.insert(0, "252")
    entry_jump_intensity.delete(0, tk.END); entry_jump_intensity.insert(0, "0.1")
    entry_jump_mean.delete(0, tk.END);      entry_jump_mean.insert(0, "0")
    entry_jump_vol.delete(0, tk.END);       entry_jump_vol.insert(0, "0.2")
    lbl_result.config(text="Your results will appear here.")
    for canvas in (sim_canvas, hist_canvas, heatmap_canvas):
        if canvas is not None:
            canvas.get_tk_widget().destroy()
    status_var.set("Inputs reset.")

def save_simulation_plot():
    """
    Saves the current simulation plot to a file.
    """
    if sim_canvas is None:
        messagebox.showerror("Error", "No simulation plot to save.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")])
    if file_path:
        sim_canvas.figure.savefig(file_path)
        status_var.set(f"Plot saved to {file_path}")

def update_jump_fields(*args):
    """
    Enables jump parameter fields only when Jump Diffusion is selected.
    """
    if combobox_model.get() == "Jump Diffusion":
        entry_jump_intensity.config(state="normal")
        entry_jump_mean.config(state="normal")
        entry_jump_vol.config(state="normal")
    else:
        entry_jump_intensity.config(state="disabled")
        entry_jump_mean.config(state="disabled")
        entry_jump_vol.config(state="disabled")

# ====================================
# UI Setup
# ====================================
root = tk.Tk()
root.title("Monte Carlo Option Pricer")
root.geometry("900x950")
root.configure(bg=PRIMARY_BG)

# NOTE: Using a consistent padding and border style throughout.
# Header
header = tk.Label(root, text="Monte Carlo Option Pricer", font=HEADER_FONT,
                  bg=PRIMARY_BG, fg=BUTTON_BG, pady=PAD_Y)
header.pack(pady=(15, 10))

# Top Frame 
top_frame = tk.Frame(root, bg=PRIMARY_BG)
top_frame.pack(fill=tk.X, padx=PAD_X, pady=PAD_Y)

# Input Frame (Left Side)
input_frame = tk.Frame(top_frame, bg=SECONDARY_BG, bd=2, relief="groove")
input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))

# ----- Basic Parameters -----
basic_frame = tk.LabelFrame(input_frame, text="Basic Parameters", bg=SECONDARY_BG,
                            fg=LABEL_COLOR, font=LABEL_FONT, bd=2, relief="ridge", padx=PAD_X, pady=PAD_Y)
basic_frame.pack(padx=PAD_X, pady=PAD_Y, fill=tk.X)

tk.Label(basic_frame, text="Initial Stock Price (S0):", bg=SECONDARY_BG, fg=LABEL_COLOR,
         font=LABEL_FONT).grid(row=0, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
entry_S0 = tk.Entry(basic_frame, font=ENTRY_FONT, bg=PRIMARY_BG, fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
entry_S0.insert(0, "100")
entry_S0.grid(row=0, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew")
ToolTip(entry_S0, "Enter the current stock price (e.g., 100).")

tk.Label(basic_frame, text="Strike Price (K):", bg=SECONDARY_BG, fg=LABEL_COLOR,
         font=LABEL_FONT).grid(row=1, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
entry_K = tk.Entry(basic_frame, font=ENTRY_FONT, bg=PRIMARY_BG, fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
entry_K.insert(0, "100")
entry_K.grid(row=1, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew")

tk.Label(basic_frame, text="Time to Expiration (T, years):", bg=SECONDARY_BG, fg=LABEL_COLOR,
         font=LABEL_FONT).grid(row=2, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
entry_T = tk.Entry(basic_frame, font=ENTRY_FONT, bg=PRIMARY_BG, fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
entry_T.insert(0, "1")
entry_T.grid(row=2, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew")

tk.Label(basic_frame, text="Risk-Free Rate (r, %):", bg=SECONDARY_BG, fg=LABEL_COLOR,
         font=LABEL_FONT).grid(row=3, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
entry_r = tk.Entry(basic_frame, font=ENTRY_FONT, bg=PRIMARY_BG, fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
entry_r.insert(0, "2.5")
entry_r.grid(row=3, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew")

tk.Label(basic_frame, text="Volatility (σ, %):", bg=SECONDARY_BG, fg=LABEL_COLOR,
         font=LABEL_FONT).grid(row=4, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
entry_sigma = tk.Entry(basic_frame, font=ENTRY_FONT, bg=PRIMARY_BG, fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
entry_sigma.insert(0, "20")
entry_sigma.grid(row=4, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew")

tk.Label(basic_frame, text="Option Type:", bg=SECONDARY_BG, fg=LABEL_COLOR,
         font=LABEL_FONT).grid(row=5, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
combobox_option = ttk.Combobox(basic_frame, values=["Call", "Put"], font=ENTRY_FONT, width=12)
combobox_option.set("Call")
combobox_option.grid(row=5, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew")

tk.Label(basic_frame, text="Model:", bg=SECONDARY_BG, fg=LABEL_COLOR,
         font=LABEL_FONT).grid(row=6, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
combobox_model = ttk.Combobox(basic_frame, values=["GBM", "Jump Diffusion"], font=ENTRY_FONT, width=15)
combobox_model.set("GBM")
combobox_model.grid(row=6, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew")
combobox_model.bind("<<ComboboxSelected>>", update_jump_fields)
basic_frame.columnconfigure(1, weight=1)

# ----- Advanced Parameters -----
advanced_frame = tk.LabelFrame(input_frame, text="Advanced Parameters", bg=SECONDARY_BG,
                               fg=LABEL_COLOR, font=LABEL_FONT, bd=2, relief="ridge", padx=PAD_X, pady=PAD_Y)
advanced_frame.pack(padx=PAD_X, pady=PAD_Y, fill=tk.X)

tk.Label(advanced_frame, text="Simulations (N):", bg=SECONDARY_BG, fg=LABEL_COLOR,
         font=LABEL_FONT).grid(row=0, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
spinbox_N = tk.Spinbox(advanced_frame, from_=10000, to=500000, increment=10000,
                       font=ENTRY_FONT, bg=PRIMARY_BG, fg=TEXT_COLOR, width=10)
spinbox_N.delete(0, tk.END)
spinbox_N.insert(0, "100000")
spinbox_N.grid(row=0, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew")

tk.Label(advanced_frame, text="Display Paths:", bg=SECONDARY_BG, fg=LABEL_COLOR,
         font=LABEL_FONT).grid(row=1, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
spinbox_paths = tk.Spinbox(advanced_frame, from_=1, to=20,
                           font=ENTRY_FONT, bg=PRIMARY_BG, fg=TEXT_COLOR, width=10)
spinbox_paths.delete(0, tk.END)
spinbox_paths.insert(0, "10")
spinbox_paths.grid(row=1, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew")

tk.Label(advanced_frame, text="Simulation Steps:", bg=SECONDARY_BG, fg=LABEL_COLOR,
         font=LABEL_FONT).grid(row=2, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
spinbox_steps = tk.Spinbox(advanced_frame, from_=50, to=500, increment=50,
                           font=ENTRY_FONT, bg=PRIMARY_BG, fg=TEXT_COLOR, width=10)
spinbox_steps.delete(0, tk.END)
spinbox_steps.insert(0, "252")
spinbox_steps.grid(row=2, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew")

tk.Label(advanced_frame, text="Jump Intensity (λ):", bg=SECONDARY_BG, fg=LABEL_COLOR,
         font=LABEL_FONT).grid(row=3, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
entry_jump_intensity = tk.Entry(advanced_frame, font=ENTRY_FONT, bg=PRIMARY_BG, fg=TEXT_COLOR,
                                insertbackground=TEXT_COLOR, state="disabled")
entry_jump_intensity.insert(0, "0.1")
entry_jump_intensity.grid(row=3, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew")

tk.Label(advanced_frame, text="Jump Mean (μ_jump):", bg=SECONDARY_BG, fg=LABEL_COLOR,
         font=LABEL_FONT).grid(row=4, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
entry_jump_mean = tk.Entry(advanced_frame, font=ENTRY_FONT, bg=PRIMARY_BG, fg=TEXT_COLOR,
                           insertbackground=TEXT_COLOR, state="disabled")
entry_jump_mean.insert(0, "0")
entry_jump_mean.grid(row=4, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew")

tk.Label(advanced_frame, text="Jump Volatility (σ_jump):", bg=SECONDARY_BG, fg=LABEL_COLOR,
         font=LABEL_FONT).grid(row=5, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
entry_jump_vol = tk.Entry(advanced_frame, font=ENTRY_FONT, bg=PRIMARY_BG, fg=TEXT_COLOR,
                          insertbackground=TEXT_COLOR, state="disabled")
entry_jump_vol.insert(0, "0.2")
entry_jump_vol.grid(row=5, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew")

advanced_frame.columnconfigure(1, weight=1)

# -------------------------------
# Buttons Frame (Right Side)
# -------------------------------
buttons_frame = tk.Frame(top_frame, bg=PRIMARY_BG)
buttons_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=PAD_X)
btn_style = {"font": ENTRY_FONT, "fg": BUTTON_FG, "bg": BUTTON_BG,
             "activebackground": PRIMARY_BG, "bd": 0, "relief": "ridge", "width": 20, "cursor": "hand2", "pady": 5}

tk.Button(buttons_frame, text="Run Simulation", command=run_simulation, **btn_style).pack(pady=PAD_Y)
tk.Button(buttons_frame, text="Calculate Option Price", command=show_option_price, **btn_style).pack(pady=PAD_Y)
tk.Button(buttons_frame, text="Reset Inputs", command=reset_inputs, **btn_style).pack(pady=PAD_Y)

# -------------------------------
# Notebook for Output Tabs
# -------------------------------
notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True, padx=PAD_X, pady=PAD_Y)

# Simulation Tab
sim_tab = tk.Frame(notebook, bg="white")
notebook.add(sim_tab, text="Simulation")
sim_frame = tk.Frame(sim_tab, bg="white")
sim_frame.pack(fill=tk.BOTH, expand=True, padx=PAD_X, pady=PAD_Y)
sim_button_frame = tk.Frame(sim_tab, bg="white")
sim_button_frame.pack(fill=tk.X, padx=PAD_X, pady=PAD_Y)
tk.Button(sim_button_frame, text="Save Plot", command=save_simulation_plot, **btn_style).pack(side=tk.RIGHT, padx=PAD_X)

# Option Pricing Tab
price_tab = tk.Frame(notebook, bg="white")
notebook.add(price_tab, text="Option Pricing")
lbl_result = tk.Label(price_tab, text="Your results will appear here.", bg="white",
                      fg=PRIMARY_BG, font=ENTRY_FONT, justify="center", padx=PAD_X, pady=PAD_Y)
lbl_result.pack(expand=True, padx=PAD_X, pady=PAD_Y)

# Histogram Tab
hist_tab = tk.Frame(notebook, bg="white")
notebook.add(hist_tab, text="Histogram")
hist_frame = tk.Frame(hist_tab, bg="white")
hist_frame.pack(fill=tk.BOTH, expand=True, padx=PAD_X, pady=PAD_Y)

# Heatmap Tab
heatmap_tab = tk.Frame(notebook, bg="white")
notebook.add(heatmap_tab, text="Heatmap")
heatmap_frame = tk.Frame(heatmap_tab, bg="white")
heatmap_frame.pack(fill=tk.BOTH, expand=True, padx=PAD_X, pady=PAD_Y)
heatmap_button_frame = tk.Frame(heatmap_tab, bg="white")
heatmap_button_frame.pack(fill=tk.X, padx=PAD_X, pady=PAD_Y)
tk.Button(heatmap_button_frame, text="Update Heatmap", command=run_heatmap, **btn_style).pack(side=tk.RIGHT, padx=PAD_X)

# -------------------------------
# Status Bar
# -------------------------------
status_var = tk.StringVar()
status_var.set("Ready")
status_bar = tk.Label(root, textvariable=status_var, bd=1, relief=tk.SUNKEN,
                      anchor="w", bg=SECONDARY_BG, fg=TEXT_COLOR, font=("Helvetica", 9), padx=PAD_X)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Start the UI Loop
root.mainloop()
