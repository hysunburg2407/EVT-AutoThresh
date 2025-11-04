import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, Canvas
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.stats
from sklearn.linear_model import LinearRegression
import os

# Multi-file state
file_paths = []                 # list[str]
files_data = {}                 # fid -> {"df": df, "rows": int, "date_min":..., "date_max":..., "vmin":..., "vmax":..., "negatives": int, "candidates_idx": list}
file_status = {}                # fid -> "Pending"|"Reviewed"|"Skipped"|"Auto"|"Error"
current_file_id = None          # int
selected_outliers_per_file = {} # fid -> set(indexes)
file_sort_mode = {}  # fid -> 'high' | 'low' | None

def set_default_window_size(win, frac_w=0.90, frac_h=0.85, min_w=900, min_h=600):
    """Resize and center a Tk/Toplevel to a sensible default."""
    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    w = max(int(sw * frac_w), min_w)
    h = max(int(sh * frac_h), min_h)
    x = (sw - w) // 2
    y = (sh - h) // 2
    win.geometry(f"{w}x{h}+{x}+{y}")
    win.minsize(min_w, min_h)

# ------------------ Step 1: File Selection Window ------------------
df = None

def open_main_window():
    """Opens the File Selection Window and loads the dataset globally."""
    global file_path, df
    file_path = None  # Initialize file path

    def select_file():
        """Handles file selection and loads dataset(s)."""
        global file_paths, current_file_id, files_data, file_status, df
    
        paths = filedialog.askopenfilenames(
            title="Select Data Files",
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx;*.xls")]
        )
        file_paths = list(paths)
    
        # If nothing selected
        if not file_paths:
            lbl_file.config(text="No files selected")
            btn_next.config(state=tk.DISABLED)
            return
    
        # Quick scan each file (fast): read + basic stats
        files_data.clear()
        file_status.clear()
    
        for fid, path in enumerate(file_paths):
            try:
                # Load CSV first; fallback to Excel
                if path.lower().endswith(".csv"):
                    _df = pd.read_csv(path)
                else:
                    _df = pd.read_excel(path)
    
                # Standardize columns the same way the app expects
                _df["date"] = pd.to_datetime(_df["date"], errors="coerce")
                _df["value"] = pd.to_numeric(_df["value"], errors="coerce")
                _df = _df.drop(columns=["Unnamed: 2"], errors="ignore")
                _df = _df.dropna(subset=["date", "value"])
    
                rows = len(_df)
                date_min, date_max = _df["date"].min(), _df["date"].max()
                vmin, vmax = _df["value"].min(), _df["value"].max()
                negatives = int((_df["value"] < 0).sum())
    
                # Heuristic for per-file candidates (kept internal; not shown in summary)
                num_years = _df["date"].dt.year.nunique()
                candidates_idx = _df.nlargest(num_years, "value").index.tolist()
    
                files_data[fid] = {
                    "df": _df,
                    "rows": rows,
                    "date_min": date_min, "date_max": date_max,
                    "vmin": vmin, "vmax": vmax,
                    "negatives": negatives,
                    "candidates_idx": candidates_idx
                }
                file_status[fid] = "Pending"
            except Exception:
                file_status[fid] = "Error"
    
        # Set context to the first file so the next window has something to show
        current_file_id = 0
        df = files_data[current_file_id]["df"].copy()
    
        # UI feedback
        lbl_file.config(text=f"Selected: {len(file_paths)} files")
        btn_next.config(state=tk.NORMAL)

    def go_to_next():
        """Closes file selection window and opens Missing Values Handling Window."""
        root.destroy()
        open_nan_handling_window()

    root = tk.Tk()
    root.title("Threshold Selection Tool: File Selection")
    root.geometry("400x200")

    lbl_file = tk.Label(root, text="Select a CSV file to process", font=("Arial", 12))
    lbl_file.pack(pady=10)

    btn_select = tk.Button(root, text="Select Input File", command=select_file)
    btn_select.pack(pady=5)

    btn_next = tk.Button(root, text="Next", command=go_to_next, state=tk.DISABLED)
    btn_next.pack(pady=10)

    root.mainloop()

# ------------------ Step 2: Missing Values Handling Window ------------------
selected_nan_methods = set()

def open_nan_handling_window():
    """Opens the Missing Values Handling Window and applies selected NaN handling to the dataset."""
    global df, selected_nan_methods
    nan_window = tk.Tk()
    nan_window.title("Threshold Selection Tool: Handle Missing Values")
    nan_window.geometry("500x400")

    nan_options = {
        "Remove rows having NaNs": "1",
        "Fill with the COLUMN MEAN": "2",
        "Fill with the COLUMN MEDIAN": "3",
        "Fill with ZERO (0)": "4",
        "Interpolate and fill using NEIGHBOURING VALUES": "5",
    }

    def handle_nan():
        """Applies selected NaN-handling method to the dataset."""
        global df, selected_nan_methods
        selected_nan_methods.clear()  # ✅ Ensure it’s updated globally
    
        for key, value in nan_options.items():
            if var_dict[value].get() == 1:
                selected_nan_methods.add(value)  # ✅ Store selected method in the global set
    
        print("✅ Selected NaN Handling Methods:", selected_nan_methods)  # ✅ Debugging Step
    
        if not selected_nan_methods:
            messagebox.showwarning("No Selection", "Please select at least one method to handle NaN values.")
            return
    
        # ✅ Convert "value" column to numeric (Fix for TypeError)
        df["value"] = pd.to_numeric(df["value"], errors='coerce')
    
        # ✅ Apply NaN handling for existing NaNs
        for method in selected_nan_methods:
            if method == "1":
                df.dropna(subset=["value"], inplace=True)
            elif method == "2":
                df["value"].fillna(df["value"].mean(), inplace=True)
            elif method == "3":
                df["value"].fillna(df["value"].median(), inplace=True)
            elif method == "4":
                df["value"].fillna(0, inplace=True)
            elif method == "5":
                df["value"] = df["value"].interpolate(method='linear', limit_direction='both')
    
        messagebox.showinfo("NaN Handling", f"✅ NaN values processed using: {', '.join([key for key, value in nan_options.items() if value in selected_nan_methods])}")
        btn_next.config(state=tk.NORMAL)

    lbl_nan = tk.Label(nan_window, text="How do you want to handle missing values (NaNs)?", font=("Arial", 12))
    lbl_nan.pack(pady=5)

    frame_nan = tk.Frame(nan_window)
    frame_nan.pack()

    var_dict = {value: tk.IntVar() for value in nan_options.values()}
    for option, value in nan_options.items():
        chk = tk.Checkbutton(frame_nan, text=option, variable=var_dict[value], font=("Arial", 10))
        chk.pack(anchor="center", pady=2)

    btn_apply_nan = tk.Button(nan_window, text="Apply NaN Handling Method", command=handle_nan)
    btn_apply_nan.pack(pady=10)

    btn_back = tk.Button(nan_window, text="Back", command=lambda: [nan_window.destroy(), open_main_window()])
    btn_back.pack(side="left", padx=20, pady=10)

    btn_next = tk.Button(nan_window, text="Next", command=lambda: [nan_window.destroy(), open_outlier_window()], state=tk.DISABLED)
    btn_next.pack(side="right", padx=20, pady=10)

    nan_window.mainloop()


# ------------------ Step 3: Outliers Selection Window ------------------
# Store outlier selections
selected_outliers = set()
# --- Analysis results (multi-file) ---
analysis_results_per_file = {}   # { fid: results_df }
analysis_file_names = {}         # { fid: basename }

def open_outlier_window():
    """Opens the Outliers Selection Window and applies outlier detection to the dataset."""
    global df, selected_outliers, files_data, file_paths, file_status, current_file_id, selected_outliers_per_file, file_sort_mode
    # runtime state for the currently selected file
    current_flagged_outliers_df = None   # pandas DataFrame of flagged outliers now shown in the list
    current_outlier_index_order = []     # list of df indices in the exact order displayed in the listbox (after sorting)
    current_total_outliers = 0           # len(current_flagged_outliers_df)

    
    outlier_window = tk.Tk()
    set_default_window_size(outlier_window, 0.92, 0.88, 1100, 720)
    outlier_window.title("Threshold Selection Tool: Outlier Processing")
    outlier_window.geometry("700x600")

    lbl_instruction = tk.Label(
        outlier_window,
        text="The values listed below are identified as potential outliers.\n\n"
             "Please select only those outliers that are likely due to **instrument errors, recording mistakes, or human input errors**.\n\n"
             "Selected outliers will be treated as NaNs and processed based on the **NaN-handling method chosen above**.\n\n"
             "If the displayed values are NOT to be considered as outliers, then skip and move to the next step.\n\n"
             "Negatiive Values (if found) will be added to selected outliers list and processed with them.",
        font=("Arial", 10),
        wraplength=500,
        justify="left"
    )
    lbl_instruction.pack(pady=10)
    
    # === Files Summary (NEW) ===
    summary_cols = ("#", "file", "rows", "daterange", "min", "max", "negs", "selected")
    files_summary = ttk.Treeview(outlier_window, columns=summary_cols, show="headings", height=6)
    
    files_summary.heading("#", text="#");            files_summary.column("#", width=40, anchor="w")
    files_summary.heading("file", text="File name"); files_summary.column("file", width=300, anchor="w")
    files_summary.heading("rows", text="Rows");      files_summary.column("rows", width=80, anchor="w")
    files_summary.heading("daterange", text="Date range"); files_summary.column("daterange", width=170, anchor="w")
    files_summary.heading("min", text="Min");        files_summary.column("min", width=90, anchor="w")
    files_summary.heading("max", text="Max");        files_summary.column("max", width=90, anchor="w")
    files_summary.heading("negs", text="Negatives"); files_summary.column("negs", width=95, anchor="w")
    files_summary.heading("selected", text="Selected");  files_summary.column("selected", width=90, anchor="w")
    
    for fid, path in enumerate(file_paths):
        meta = files_data.get(fid)
        if not meta:
            values = (fid + 1, os.path.basename(path), "-", "-", "-", "-", "-", 0)
        else:
            rng = f"{meta['date_min']:%Y-%m-%d} — {meta['date_max']:%Y-%m-%d}"
            sel_count = len(selected_outliers_per_file.get(fid, set()))
            values = (
                fid + 1,
                os.path.basename(path),
                meta["rows"],
                rng,
                round(meta["vmin"], 3),
                round(meta["vmax"], 3),
                meta["negatives"],
                sel_count
            )
        files_summary.insert("", "end", iid=str(fid), values=values)
    
    files_summary.pack(fill="x", padx=10, pady=(8, 8))
    # === End Files Summary ===

    frame_outliers = tk.Frame(outlier_window)
    frame_outliers.pack(pady=10)
    frame_outliers.grid_columnconfigure(0, weight=0)
    frame_outliers.grid_columnconfigure(1, weight=0)
    frame_outliers.grid_columnconfigure(2, minsize=180)  # space for sort panel
    
    lbl_outlier = tk.Label(frame_outliers, text="Select Outliers to Process", font=("Arial", 12))
    lbl_outlier.grid(row=0, column=0, padx=10, columnspan=2)

    scrollbar_outliers = tk.Scrollbar(frame_outliers, orient="vertical")
    outlier_list = tk.Listbox(frame_outliers, selectmode=tk.MULTIPLE, height=6, width=40, yscrollcommand=scrollbar_outliers.set)
    outlier_list.grid(row=1, column=0, padx=10)
    scrollbar_outliers.config(command=outlier_list.yview)
    scrollbar_outliers.grid(row=1, column=1, sticky="ns")
    
    # ------------------ "Select All" and "Deselect All" Buttons ------------------
    def select_all_outliers():
        """Selects all outlier values, skipping headers, and updates selected_outliers."""
        if outlier_list.size() > 2:  # Ensure list has actual outliers
            outlier_list.select_set(2, tk.END)  # ✅ Skip first two rows (headers & separator)
    
            # ✅ Manually update selected_outliers set
            global selected_outliers
            selected_outliers = {outlier_list.get(i) for i in range(2, outlier_list.size())}
    
            # ✅ Ensure "Process Selected Outliers" button activates
            enable_process_button()

    def deselect_all_outliers():
        """Deselects all outlier values, keeping headers visible."""
        if outlier_list.size() > 2:  # Ensure list has actual outliers
            outlier_list.selection_clear(2, tk.END)  # ✅ Skip first two rows (headers & separator)

            # ✅ Trigger the same logic as manual selection
            enable_process_button(None)  # Call the function to update selected_outliers
    frame_select_buttons = tk.Frame(frame_outliers)
    frame_select_buttons.grid(row=2, column=0, columnspan=2, pady=5)

    btn_select_all = tk.Button(frame_select_buttons, text="Select All", command=select_all_outliers)
    btn_deselect_all = tk.Button(frame_select_buttons, text="Deselect All", command=deselect_all_outliers)

    btn_select_all.grid(row=0, column=0, padx=5)
    btn_deselect_all.grid(row=0, column=1, padx=5)


    # ------------------ Detect & Flag Outliers ------------------
    df["Flag"] = "Normal"

    # ✅ Flag Negative Values (Already Exists)
    df.loc[df["value"] < 0, "Flag"] = "Negative Value (Erroneous)"

    # ✅ Flag Top n Extreme Values (Already Exists)
    num_years = df['date'].dt.year.nunique()
    top_n_values = df.nlargest(num_years, "value")
    df.loc[top_n_values.index, "Flag"] = "Potential Outlier (User Review)"

    # ✅ Count total flagged outliers FIRST before using it
    total_outliers = df[df["Flag"] == "Potential Outlier (User Review)"].shape[0]

    # ✅ Store flagged outliers
    flagged_outliers = df[df["Flag"] == "Potential Outlier (User Review)"]


    # ✅ Count Negative Values
    num_negative_values = df[df["Flag"] == "Negative Value (Erroneous)"].shape[0]

    # ✅ Display Negative Values Count Above Listbox
    lbl_neg_values = tk.Label(
        outlier_window,
        text=f"Number of negative values found: {num_negative_values}" if num_negative_values > 0 else "✅ No negative values detected.",
        font=("Arial", 12),
        fg="blue" if num_negative_values > 0 else "green"
    )
    lbl_neg_values.pack(pady=5)
    
    # ✅ Display Selected Outliers Counter (Live Update)
    lbl_selected_outliers = tk.Label(
        outlier_window,
        text="Selected Outliers: 0 / 0",  # Initially 0 selected out of 0
        font=("Arial", 12),
        fg="blue"
    )
    lbl_selected_outliers.pack(pady=5)
    
    # ------------------ Ensure Buttons Are Always Visible ------------------
    frame_buttons = tk.Frame(outlier_window)
    frame_buttons.pack(pady=10)

    def apply_nan_handling():
        """Applies the selected NaN-handling method to the dataset after processing outliers."""
        global df, selected_nan_methods

        if not selected_nan_methods:
            return  # If no method was selected, do nothing

        # ✅ Convert "value" column to numeric (Fix for TypeError)
        df["value"] = pd.to_numeric(df["value"], errors='coerce')

        # ✅ Apply the NaN-handling methods selected in Step 2
        for method in selected_nan_methods:
            if method == "1":
                df.dropna(subset=["value"], inplace=True)
            elif method == "2":
                df["value"].fillna(df["value"].mean(), inplace=True)
            elif method == "3":
                df["value"].fillna(df["value"].median(), inplace=True)
            elif method == "4":
                df["value"].fillna(0, inplace=True)
            elif method == "5":
                df["value"] = df["value"].interpolate(method='linear', limit_direction='both')  # ✅ Ensure both directions are filled

    def apply_nan_handling_df(local_df):
        """Apply the selected NaN-handling method to the given DataFrame and return it."""
        global selected_nan_methods
        if not selected_nan_methods:
            return local_df
    
        local_df["value"] = pd.to_numeric(local_df["value"], errors='coerce')
    
        for method in selected_nan_methods:
            if method == "1":
                local_df.dropna(subset=["value"], inplace=True)
            elif method == "2":
                local_df["value"].fillna(local_df["value"].mean(), inplace=True)
            elif method == "3":
                local_df["value"].fillna(local_df["value"].median(), inplace=True)
            elif method == "4":
                local_df["value"].fillna(0, inplace=True)
            elif method == "5":
                local_df["value"] = local_df["value"].interpolate(method='linear', limit_direction='both')
        return local_df


    def process_outliers():
        """Process selected outliers for ALL files that have selections, in one go."""
        global df
    
        # Collect files that have any manual selection
        files_with_selection = [(fid, idxs) for fid, idxs in selected_outliers_per_file.items() if idxs]
        if not files_with_selection:
            messagebox.showwarning("No Selection", "⚠️ No files have selected outliers.")
            return
    
        # One dialog for all files: Remove rows vs Process as NaNs
        def custom_outlier_choice():
            custom_dialog = tk.Toplevel()
            custom_dialog.title("Outlier Processing")
            custom_dialog.geometry("400x220")
    
            lbl_message = tk.Label(
                custom_dialog,
                text=("Apply to ALL files with selected outliers:\n\n"
                      "• Remove selected outlier rows (+ negatives)\n"
                      "• OR set them to NaN (then apply the chosen NaN-handling)"),
                font=("Arial", 10),
                wraplength=360,
                justify="left"
            )
            lbl_message.pack(pady=20)
    
            choice_var = tk.StringVar()
    
            def remove_outliers():
                choice_var.set("remove")
                custom_dialog.destroy()
    
            def process_as_nan():
                choice_var.set("nan")
                custom_dialog.destroy()
    
            tk.Button(custom_dialog, text="Remove Outlier Row(s)", command=remove_outliers).pack(pady=5)
            tk.Button(custom_dialog, text="Process as NaNs", command=process_as_nan).pack(pady=5)
    
            custom_dialog.wait_window()
            return choice_var.get()
    
        choice = custom_outlier_choice()
        if choice not in ("remove", "nan"):
            return
    
        processed_files = 0
    
        # Loop over each file that has a non-empty selection
        for fid, selected_idx_set in files_with_selection:
            # Work on a copy of that file's dataframe
            df_local = files_data[fid]["df"].copy()
    
            # Rebuild flags to capture negatives for this file
            df_local["Flag"] = "Normal"
            df_local.loc[df_local["value"] < 0, "Flag"] = "Negative Value (Erroneous)"
    
            # Always include NEGATIVES along with the user's selections
            negative_value_indexes = df_local[df_local["Flag"] == "Negative Value (Erroneous)"].index.tolist()
            combined_indexes = list(set(list(selected_idx_set) + negative_value_indexes))
    
            if not combined_indexes:
                # Nothing to do for this file; continue to next
                continue
    
            if choice == "remove":
                df_local.drop(index=combined_indexes, inplace=True)
            else:
                # Set to NaN, then apply the selected NaN-handling method
                df_local.loc[combined_indexes, "value"] = np.nan
                df_local = apply_nan_handling_df(df_local)
    
            # Save cleaned df back
            files_data[fid]["df"] = df_local
    
            # Clear selection for this file
            selected_outliers_per_file[fid] = set()
    
            # Update the Files Summary "Selected" count to 0
            vals = list(files_summary.item(str(fid), "values"))
            if vals:
                vals[-1] = 0
                files_summary.item(str(fid), values=tuple(vals))
    
            processed_files += 1
    
        # Refresh the CURRENT file's view (so the list reflects changes immediately)
        build_flags_for_file(current_file_id)
    
        # Enable Next and inform the user
        btn_next.config(state=tk.NORMAL)
        messagebox.showinfo("Outlier Processing",
                            f"✅ Processed outliers in {processed_files} file(s). Selections cleared.")


    btn_process = tk.Button(frame_buttons, text="Process Selected Outliers", command=process_outliers, state=tk.DISABLED)
    
    def process_negative_values_only():
        """Processes negative values as NaN and applies the selected NaN-handling method before skipping."""
        global df
        
        # ✅ Check if there are any invalid entries (negative + non-numeric)
        invalid_value_indexes = df[df["Flag"].str.contains("Erroneous", na=False)].index.tolist()
        
        if invalid_value_indexes:
            # ✅ Convert all invalid values (negative + non-numeric) to NaN
            df.loc[invalid_value_indexes, "value"] = np.nan
            
            # ✅ Apply the same NaN-handling method chosen earlier
            apply_nan_handling()
            
            # ✅ Log and show message
            print("Negative values processed before skipping outlier selection.")
            messagebox.showinfo("Negative Values Processed", "✅ Negative values were processed and handled before skipping.")
        
        # Mark as Skipped (no manual outliers selected)
        fid = current_file_id
        selected_outliers_per_file[fid] = set()
        file_status[fid] = "Skipped"
        
        # Update the Status cell in the Files Summary table
        vals = list(files_summary.item(str(fid), "values"))
        vals[-1] = "Skipped"
        files_summary.item(str(fid), values=tuple(vals))

        # ✅ Move to the next step
        outlier_window.destroy()
        open_save_data_window()
    
    btn_skip = tk.Button(frame_buttons, text="Skip", command=process_negative_values_only)
    btn_back = tk.Button(frame_buttons, text="Back", command=lambda: [outlier_window.destroy(), open_nan_handling_window()])

    btn_process.grid(row=0, column=0, padx=5)
    btn_skip.grid(row=0, column=1, padx=5)
    btn_back.grid(row=0, column=2, padx=5)
    btn_next = tk.Button(frame_buttons, text="Next", state=tk.DISABLED,
                     command=lambda: [outlier_window.destroy(), open_save_data_window()])
    btn_next.grid(row=0, column=3, padx=5)

    
    # ✅ Ensure the listbox gets populated
    def update_outlier_list(sorted_df):
        """Updates the outlier listbox based on sorting and records current order."""
        nonlocal current_flagged_outliers_df, current_outlier_index_order, current_total_outliers
    
        # record the current df and order for later (sorting / processing)
        current_flagged_outliers_df = sorted_df
        current_outlier_index_order = list(sorted_df.index)
        current_total_outliers = len(sorted_df)
    
        outlier_list.delete(0, tk.END)
        # headers
        outlier_list.insert(tk.END, "Date               Value")
        outlier_list.insert(tk.END, "-" * 30)
    
        for _, row in sorted_df.iterrows():
            outlier_list.insert(tk.END, f"{row['date']}       {row['value']:.2f}")
    
        # reset the selected counter line and button state
        lbl_selected_outliers.config(text=f"Selected Outliers: 0 / {current_total_outliers}")
        btn_process.config(state=(tk.NORMAL if current_total_outliers > 0 else tk.DISABLED))

    def build_flags_for_file(fid: int):
        """Switch to file `fid`, rebuild flags, and refresh the outlier list & counters."""
        global df, current_file_id
    
        current_file_id = fid
        df = files_data[fid]["df"].copy()
    
        # Flagging (same logic you already use)
        df["Flag"] = "Normal"
        df.loc[df["value"] < 0, "Flag"] = "Negative Value (Erroneous)"
        num_years_local = df['date'].dt.year.nunique()
        top_n_values_local = df.nlargest(num_years_local, "value")
        df.loc[top_n_values_local.index, "Flag"] = "Potential Outlier (User Review)"
    
        # Recompute for this file
        flagged_outliers_local = df[df["Flag"] == "Potential Outlier (User Review)"]
        total_outliers_local = flagged_outliers_local.shape[0]
        num_negative_values_local = df[df["Flag"] == "Negative Value (Erroneous)"].shape[0]
        # Apply the last-used sort mode for this file (if any)
        mode = file_sort_mode.get(fid)
        if mode == 'high':
            sorted_df_local = flagged_outliers_local.sort_values(by="value", ascending=False)
        elif mode == 'low':
            sorted_df_local = flagged_outliers_local.sort_values(by="value", ascending=True)
        else:
            sorted_df_local = flagged_outliers_local
        
        # Update the negative-values label (reuse your styling)
        lbl_neg_values.config(
            text=(f"Number of negative values found: {num_negative_values_local}"
                  if num_negative_values_local > 0 else "✅ No negative values detected."),
            fg=("blue" if num_negative_values_local > 0 else "green")
        )
    
        # Refresh the listbox with headers using your helper
        if flagged_outliers_local.empty:
            outlier_list.delete(0, tk.END)
            outlier_list.insert(tk.END, "✅ No potential outliers detected.")
            lbl_selected_outliers.config(text="Selected Outliers: 0 / 0")
            btn_process.config(state=tk.DISABLED)
        else:
            # Use the per-file sorted view (or unsorted if no mode set)
            update_outlier_list(sorted_df_local)
            lbl_selected_outliers.config(text=f"Selected Outliers: 0 / {total_outliers_local}")
            btn_process.config(state=tk.DISABLED)

        # === Restore previous selection for this file (if any) ===
        prev = selected_outliers_per_file.get(fid, set())

        # Only attempt to map positions if we currently have a displayed list order
        if current_outlier_index_order:
            # Map df index -> listbox position (+2 for header rows)
            pos_by_index = {idx: pos for pos, idx in enumerate(current_outlier_index_order, start=2)}
            to_select = [pos_by_index[idx] for idx in prev if idx in pos_by_index]

            # Apply selection in the listbox
            for pos in to_select:
                try:
                    outlier_list.selection_set(pos)
                except Exception:
                    pass

            # Update counter and Process button based on restored selection
            lbl_selected_outliers.config(text=f"✔️ Selected Outliers: {len(prev)} / {current_total_outliers}")
            btn_process.config(state=(tk.NORMAL if prev else tk.DISABLED))


    def save_current_selection():
        """Save the current file's selected outliers (by DataFrame index) + update table count."""
        # No list yet (first open)
        if not current_outlier_index_order:
            return
    
        # What is currently highlighted in the listbox (skip the 2 header rows)
        sel_positions = outlier_list.curselection()
        chosen = {current_outlier_index_order[i - 2] for i in sel_positions if i >= 2}
    
        # Save to memory for this file
        selected_outliers_per_file[current_file_id] = chosen
    
        # Update the 'Selected' count in the Files Summary table for this row
        vals = list(files_summary.item(str(current_file_id), "values"))
        vals[-1] = len(chosen)
        files_summary.item(str(current_file_id), values=tuple(vals))

    def on_summary_row_select(event=None):
        sel = files_summary.selection()
        if not sel:
            return
        # save current file's selection before leaving it
        save_current_selection()
    
        fid = int(sel[0])  # we used fid as the Treeview item's iid
        build_flags_for_file(fid)

    files_summary.bind("<<TreeviewSelect>>", on_summary_row_select)

    # ✅ Fix "Bad Listbox Index" Error When Restoring Selections
    for idx in selected_outliers:
        try:
            outlier_list.selection_set(outlier_list.get(0, tk.END).index(str(idx)))  # ✅ Restore selections properly
        except ValueError:
            pass  # Ignore if index is not found

    # ------------------ Sorting Options (Beside Listbox) ------------------
    frame_sort = tk.Frame(frame_outliers)
    frame_sort.grid(row=1, column=2, padx=20, sticky="n")

    lbl_sort = tk.Label(frame_sort, text="Sort Outliers By:", font=("Arial", 12))
    lbl_sort.pack()

    def sort_high_to_low():
        # Inside sort_high_to_low()
        if current_flagged_outliers_df is None or current_flagged_outliers_df.empty:
            return
        file_sort_mode[current_file_id] = 'high'
        sorted_df = current_flagged_outliers_df.sort_values(by="value", ascending=False)
        update_outlier_list(sorted_df)

    def sort_low_to_high():
        # Inside sort_low_to_high()
        if current_flagged_outliers_df is None or current_flagged_outliers_df.empty:
            return
        file_sort_mode[current_file_id] = 'low'
        sorted_df = current_flagged_outliers_df.sort_values(by="value", ascending=True)
        update_outlier_list(sorted_df)

    btn_sort_high = tk.Button(frame_sort, text="Value (High to Low)", command=sort_high_to_low)
    btn_sort_low = tk.Button(frame_sort, text="Value (Low to High)", command=sort_low_to_high)

    btn_sort_high.pack(pady=2)
    btn_sort_low.pack(pady=2)

    # ------------------ Enable "Process Selected Outliers" When Outliers Are Selected ------------------
    def enable_process_button(event=None):
        """Updates selected outliers count and enables/disables the process button."""
        global selected_outliers
        selected_items = outlier_list.curselection()
    
        # ✅ Ignore header rows and select only valid outliers
        selected_outliers = {outlier_list.get(i) for i in selected_items if i >= 2}
        
        # Persist the selection for the current file (by df index, not display text)
        selected_outliers_per_file[current_file_id] = {
            current_outlier_index_order[i - 2] for i in outlier_list.curselection() if i >= 2
        }
        
        # Update the 'Selected' count in the Files Summary table for this file
        vals = list(files_summary.item(str(current_file_id), "values"))
        vals[-1] = len(selected_outliers_per_file[current_file_id])
        files_summary.item(str(current_file_id), values=tuple(vals))

        # ✅ Enable button only if at least one valid outlier is selected
        btn_process.config(state=tk.NORMAL if selected_outliers else tk.DISABLED)
    
        # ✅ Update Selected Outliers Counter (use the persisted count)
        num_selected_outliers = len(selected_outliers_per_file[current_file_id])
        lbl_selected_outliers.config(
            text=f"✔️ Selected Outliers: {num_selected_outliers} / {current_total_outliers}"
        )

    outlier_list.bind("<<ListboxSelect>>", enable_process_button)
    
    def prevent_header_selection(event):
        """Completely prevents selection of headers & separator (removes highlight too)."""
        selected_items = outlier_list.curselection()

        # ✅ If any invalid selections (header or separator), clear selection
        if any(i < 2 for i in selected_items):
            outlier_list.selection_clear(0, 1)  # ✅ Clear headers & separator line
            return "break"  # ✅ Stops Tkinter from processing selection


    outlier_list.bind("<ButtonPress-1>", prevent_header_selection)  # ✅ Stops selection BEFORE it happens
    outlier_list.bind("<<ListboxSelect>>", enable_process_button)  # ✅ Ensures button activates properly

    # Select the first file by default and build its outlier list
    if file_paths:
        files_summary.selection_set("0")
        files_summary.focus("0")
        build_flags_for_file(0)

    outlier_window.mainloop()

# ------------------ Step 4: Save Cleaned Data Window ------------------
# Store save selection
save_selected = False

def open_save_data_window():
    """Opens the Save Cleaned Data Window and saves ALL cleaned files to a chosen folder as *_cleaned.csv."""
    global df, file_paths, files_data

    save_window = tk.Tk()
    set_default_window_size(save_window, 0.65, 0.55, 700, 450)
    save_window.title("Threshold Selection Tool: Save Data")
    save_window.geometry("560x320")

    # ---- Intro text ----
    lbl_save_prompt = tk.Label(
        save_window,
        text=("Your data has been cleaned (outliers/negatives handled).\n\n"
              "Choose a folder and I will save ALL cleaned files there\n"
              "with '_cleaned' added to their original names."),
        font=("Arial", 10),
        wraplength=520,
        justify="left"
    )
    lbl_save_prompt.pack(pady=10)

    # ---- Default folder = first file's folder (if available) ----
    default_dir = ""
    try:
        if file_paths and len(file_paths) > 0:
            default_dir = os.path.dirname(file_paths[0]) or ""
    except Exception:
        default_dir = ""

    chosen_folder = tk.StringVar(value=default_dir)

    # ---- Folder chooser row ----
    folder_frame = tk.Frame(save_window)
    folder_frame.pack(fill="x", padx=12, pady=(0, 8))

    tk.Label(folder_frame, text="Save to folder:", font=("Arial", 10)).grid(row=0, column=0, sticky="w", padx=(0, 8))
    folder_entry = tk.Entry(folder_frame, textvariable=chosen_folder, width=52, state="readonly")
    folder_entry.grid(row=0, column=1, sticky="w")

    def pick_folder():
        sel = filedialog.askdirectory(initialdir=chosen_folder.get() or os.getcwd(), title="Choose output folder")
        if sel:
            chosen_folder.set(sel)

    tk.Button(folder_frame, text="Choose Folder…", command=pick_folder).grid(row=0, column=2, padx=8)

    # ---- Save logic ----
    def save_all_cleaned():
        out_dir = chosen_folder.get().strip()
        if not out_dir:
            messagebox.showwarning("Choose Folder", "Please choose a folder to save the cleaned files.")
            return
        if not os.path.isdir(out_dir):
            messagebox.showerror("Invalid Folder", f"The folder does not exist:\n{out_dir}")
            return

        total = 0
        failures = []

        # Iterate over all loaded files in order
        for fid, src_path in enumerate(file_paths):
            try:
                if fid not in files_data or "df" not in files_data[fid]:
                    continue  # nothing to save for this id

                df_local = files_data[fid]["df"].copy()

                # Ensure date column is a clean string YYYY-MM-DD if present
                if "date" in df_local.columns:
                    df_local["date"] = pd.to_datetime(df_local["date"], errors="coerce").dt.strftime("%Y-%m-%d")

                # Drop internal working column(s) from the output
                if "Flag" in df_local.columns:
                    df_local = df_local.drop(columns=["Flag"])
                base = os.path.splitext(os.path.basename(src_path))[0]
                out_name = f"{base}_cleaned.csv"
                out_path = os.path.join(out_dir, out_name)

                # Write CSV without index
                df_local.to_csv(out_path, index=False)
                total += 1
            except Exception as e:
                failures.append((os.path.basename(src_path), str(e)))

        if failures:
            msg = [f"Saved {total} file(s). Some files failed:"]
            for name, err in failures:
                msg.append(f"  • {name}: {err}")
            messagebox.showwarning("Save Completed with Warnings", "\n".join(msg))
        else:
            messagebox.showinfo("Save Successful", f"✅ Saved {total} cleaned file(s) to:\n{out_dir}")

    # ---- Buttons ----
    frame_save_buttons = tk.Frame(save_window)
    frame_save_buttons.pack(pady=10)

    btn_save_all = tk.Button(frame_save_buttons, text="Save All (_cleaned.csv)", command=save_all_cleaned)
    btn_next_step = tk.Button(frame_save_buttons, text="Move to Next Step",
                              command=lambda: [save_window.destroy(), open_analysis_intro_window()])

    btn_save_all.grid(row=0, column=0, padx=6)
    btn_next_step.grid(row=0, column=1, padx=6)

    # Back keeps your current outlier window state when reopened
    btn_back = tk.Button(save_window, text="Back",
                         command=lambda: [save_window.destroy(), open_outlier_window()])
    btn_back.pack(pady=6)

    save_window.mainloop()


#-----------------DATA ANALYSIS------------------------------------------------    
    
def open_analysis_intro_window():
    """Data Overview: left panel lists files; right pane previews the selected file (plot + stats).
       Start Analysis will pass ALL cleaned files forward (we'll handle analysis later)."""
    global files_data, file_paths

    # --------- Window & layout ---------
    analysis_intro_window = tk.Tk()
    set_default_window_size(analysis_intro_window, 0.90, 0.80, 1000, 650)
    analysis_intro_window.title("Threshold Selection Tool: Data Overview")
    analysis_intro_window.geometry("900x600")

    # Split: left (files), right (preview)
    root_frame = tk.Frame(analysis_intro_window)
    root_frame.pack(fill="both", expand=True)

    left_panel = tk.Frame(root_frame, width=240)
    left_panel.pack(side="left", fill="y")
    right_panel = tk.Frame(root_frame)
    right_panel.pack(side="right", fill="both", expand=True)

    # --------- Left: Files list ---------
    tk.Label(left_panel, text="Files", font=("Arial", 12, "bold")).pack(pady=(10, 4))

    files_list = tk.Listbox(left_panel, height=20)
    files_list.pack(fill="y", expand=False, padx=10, pady=(0, 10))

    # Populate with basenames
    for p in file_paths:
        files_list.insert(tk.END, os.path.basename(p))

    # --------- Right: existing preview UI (plot + stats + button) ---------
    # Container frames for plot and stats so we can refresh them
    plot_frame = tk.Frame(right_panel)
    plot_frame.pack(pady=10, fill="x")
    stats_frame = tk.Frame(right_panel)
    stats_frame.pack(pady=10)

    # We'll store references so we can clear/replace on selection
    current_canvas = {"canvas": None}
    current_fig = {"fig": None}

    # Labels for stats (3 columns, like before)
    lbl_col1 = tk.Label(stats_frame, font=("Arial", 12), justify="left", padx=20)
    lbl_col2 = tk.Label(stats_frame, font=("Arial", 12), justify="left", padx=20)
    lbl_col3 = tk.Label(stats_frame, font=("Arial", 12), justify="left", padx=20)
    lbl_col1.grid(row=0, column=0)
    lbl_col2.grid(row=0, column=1)
    lbl_col3.grid(row=0, column=2)

    # --------- Helper: render selected file ---------
    def render_preview(fid: int):
        # Clear old plot (if any)
        if current_canvas["canvas"] is not None:
            current_canvas["canvas"].get_tk_widget().destroy()
            current_canvas["canvas"] = None
        if current_fig["fig"] is not None:
            plt.close(current_fig["fig"])
            current_fig["fig"] = None

        # Get cleaned df for this file
        if fid not in files_data or "df" not in files_data[fid]:
            # No data available
            for lbl in (lbl_col1, lbl_col2, lbl_col3):
                lbl.config(text="—")
            note = tk.Label(plot_frame, text="No data available for this file.", font=("Arial", 11))
            note.pack()
            return

        df_local = files_data[fid]["df"].copy()

        # Build plot ONLY if both columns exist and are non-empty
        have_plot = False
        if "date" in df_local.columns and "value" in df_local.columns and len(df_local) > 0:
            try:
                # Plot
                fig, ax = plt.subplots(figsize=(6.5, 3.2))
                ax.plot(df_local["date"], df_local["value"], label="Time Series")
                ax.set_title("Time Series of Cleaned Data")
                ax.set_xlabel("Date")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
                fig.tight_layout()

                canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                canvas.get_tk_widget().pack(pady=4)
                canvas.draw()

                current_fig["fig"] = fig
                current_canvas["canvas"] = canvas
                have_plot = True
            except Exception:
                have_plot = False

        if not have_plot:
            # Show a small note if plot couldn't be rendered
            note = tk.Label(plot_frame, text="No data to plot.", font=("Arial", 11))
            note.pack()

        # Stats (same layout as before; compute on 'value' if present)
        try:
            stats = df_local["value"].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
            col1_text = (f"Minimum: {stats['min']:.2f}\n"
                         f"Maximum: {stats['max']:.2f}\n"
                         f"Count: {int(stats['count'])}")
            col2_text = (f"Mean: {stats['mean']:.2f}\n"
                         f"Median: {stats['50%']:.2f}\n"
                         f"Std Dev: {stats['std']:.2f}")
            col3_text = (f"25th Percentile: {stats['25%']:.2f}\n"
                         f"75th Percentile: {stats['75%']:.2f}\n"
                         f"90th Percentile: {stats['90%']:.2f}")
        except Exception:
            col1_text = "Minimum: —\nMaximum: —\nCount: —"
            col2_text = "Mean: —\nMedian: —\nStd Dev: —"
            col3_text = "25th Percentile: —\n75th Percentile: —\n90th Percentile: —"

        lbl_col1.config(text=col1_text)
        lbl_col2.config(text=col2_text)
        lbl_col3.config(text=col3_text)

    # --------- File selection handler (click to preview) ---------
    def on_select(event=None):
        sel = files_list.curselection()
        if not sel:
            return
        fid = sel[0]
        # Clear the plot area (widgets) before re-render
        for w in plot_frame.winfo_children():
            w.destroy()
        render_preview(fid)

    files_list.bind("<<ListboxSelect>>", on_select)

    # Default selection = first file
    if file_paths:
        files_list.selection_set(0)
        files_list.activate(0)
        render_preview(0)

    # --------- Start Analysis (pass ALL files forward) ---------
    # We'll pass the entire files_data; you'll update open_processing_dialog to accept it.
    btn_start_analysis = tk.Button(
        right_panel,
        text="Start Analysis",
        command=lambda: [analysis_intro_window.destroy(), open_processing_dialog(files_data)]
    )
    btn_start_analysis.pack(pady=16)

    analysis_intro_window.mainloop()


# Placeholder function for next analysis window

def open_processing_dialog(files_data):
    """Opens a processing dialog and runs perform_analysis(df) for ALL loaded files."""
    global analysis_results_per_file, analysis_file_names, file_paths

    processing_dialog = tk.Tk()
    set_default_window_size(processing_dialog, 0.40, 0.30, 500, 250)
    processing_dialog.title("Processing…")
    processing_dialog.geometry("420x160")

    lbl_message = tk.Label(processing_dialog, text="Preparing…", font=("Arial", 12), justify="left")
    lbl_message.pack(pady=20)

    processing_dialog.update_idletasks()  # show the window immediately

    # Reset/prepare containers
    analysis_results_per_file = {}
    analysis_file_names = {}

    # Loop over all files currently in memory
    total = len(file_paths)
    progress = ttk.Progressbar(processing_dialog, orient="horizontal",
                           mode="determinate", maximum=total, length=360)
    progress.pack(pady=10)

    for idx, src_path in enumerate(file_paths):
        fid = idx  # zero-based id for our dicts
        basename = os.path.basename(src_path)

        lbl_message.config(text=f"Processing {fid + 1} of {total}:\n{basename}")
        processing_dialog.update_idletasks()
        progress['value'] = fid + 1   # advance to current file count
        processing_dialog.update_idletasks()

        try:
            # Get the cleaned dataframe for this file
            if fid not in files_data or "df" not in files_data[fid]:
                # Nothing to analyze for this id
                analysis_results_per_file[fid] = pd.DataFrame()
                analysis_file_names[fid] = basename
                continue

            df_local = files_data[fid]["df"].copy()

            # Call your existing single-file analysis (UNCHANGED)
            results_df = perform_analysis(df_local)

            # Store
            analysis_results_per_file[fid] = results_df
            analysis_file_names[fid] = basename

        except Exception as e:
            # On error, store an empty results table for this file and continue
            analysis_results_per_file[fid] = pd.DataFrame()
            analysis_file_names[fid] = basename
            # Optional: console debug
            print(f"[Analysis error] {basename}: {e}")

    progress['value'] = total
    processing_dialog.update_idletasks()
    # Done – close the dialog
    processing_dialog.destroy()
    open_results_window()  # <-- open your hub (the window in your screenshot)

def perform_analysis(df):
    """Performs extreme value analysis and returns results DataFrame."""
    # Generate threshold array
    start = np.quantile(df["value"].values, 0.8)
    stop = df["value"].nlargest(10).iloc[-1]
    thresholds = np.linspace(start=start, stop=stop, num=100)
    
    # Define return period
    return_period = 100  # 100-year return period
    return_period_size = 365.2425  # Convert return period into days
    
    # Define number of bootstrap samples
    n_samples = 100  # For bootstrapping
    
    results = []
    for threshold in thresholds:
        extreme_values = df["value"].loc[df["value"] > threshold]
        if len(extreme_values) < 10:
            continue
        
        c, loc, scale = scipy.stats.genpareto.fit(extreme_values, floc=threshold)
        
        # Compute return value for the given return period using GPD
        num_exceedances = len(extreme_values)
        total_observations = len(df)
        exceedance_probability = 1 / (return_period * (num_exceedances / (total_observations / return_period_size)))
        return_value_gpd = scipy.stats.genpareto.isf(exceedance_probability, c, loc=threshold, scale=scale)
        
        # Compute return value using Exponential Distribution (when ξ ≈ 0, exponential is a special case of GPD)
        return_value_expon = scipy.stats.expon.isf(exceedance_probability, loc=threshold, scale=scale)
        
        # Compute AIC for GPD
        log_likelihood_gpd = np.sum(scipy.stats.genpareto.logpdf(extreme_values, c, loc=threshold, scale=scale))
        k_gpd = 2  # Number of estimated parameters (shape and scale)
        n = len(extreme_values)
        aic_gpd = 2 * k_gpd - 2 * log_likelihood_gpd + (2 * k_gpd**2 + 2 * k_gpd) / (n - k_gpd - 1) if n > k_gpd + 1 else np.nan
        
        #Compute AIC for GPD
        log_likelihood_expon = np.sum(scipy.stats.expon.logpdf(extreme_values, loc=threshold, scale=scale))
        k_expon = 1  # Only scale parameter
        aic_expon = 2 * k_expon - 2 * log_likelihood_expon + (2 * k_expon**2 + 2 * k_expon) / (n - k_expon - 1) if n > k_expon + 1 else np.nan
        
        results.append({"Threshold": threshold, "Shape": c, "Scale": scale, "Return value (GPD)": return_value_gpd, "Return value (ED)": return_value_expon, 
                        "AIC_GPD": aic_gpd, "AIC_ED": aic_expon})
    
    return pd.DataFrame(results)

def open_results_window():
    """Opens the results window with options to view tables and plots (multi-file)."""
    results_window = tk.Tk()
    set_default_window_size(results_window, 0.70, 0.55, 850, 550)
    results_window.title("Analysis Results")
    results_window.geometry("400x200")

    lbl_message = tk.Label(
        results_window,
        text="Analysis complete. You can now view results.",
        font=("Arial", 12)
    )
    lbl_message.pack(pady=10)

    # No args now—table/plots functions will read from globals
    btn_show_table = tk.Button(results_window, text="Show Results Table",
                               command=lambda: show_results_table())
    btn_show_plots = tk.Button(results_window, text="Show Plots",
                               command=lambda: show_plots())
    btn_next = tk.Button(
        results_window,
        text="Next",
        command=lambda: [results_window.destroy(), open_next_step()]
    )
    
    btn_show_table.pack(pady=5)
    btn_show_plots.pack(pady=5)
    btn_next.pack(pady=5)

    results_window.mainloop()

def show_results_table():
    """Multi-file Results Table: left file list; right table for selected file; save all to a folder."""
    global analysis_results_per_file, analysis_file_names, file_paths

    table_window = tk.Toplevel()
    set_default_window_size(table_window, 0.85, 0.75, 1000, 600)
    table_window.title("Results Table")
    table_window.geometry("900x560")

    root = tk.Frame(table_window)
    root.pack(fill="both", expand=True)

    # -------- Left panel: files list --------
    left = tk.Frame(root, width=240)
    left.pack(side="left", fill="y")
    tk.Label(left, text="Files", font=("Arial", 12, "bold")).pack(pady=(10, 6))
    lst_files = tk.Listbox(left, height=24)
    lst_files.pack(fill="y", expand=False, padx=10, pady=(0, 10))

    file_ids = list(range(len(file_paths)))
    for fid in file_ids:
        name = analysis_file_names.get(fid, os.path.basename(file_paths[fid]))
        lst_files.insert(tk.END, name)

    # -------- Right panel: table + save button --------
    right = tk.Frame(root)
    right.pack(side="right", fill="both", expand=True)

    table_frame = tk.Frame(right)
    table_frame.pack(fill="both", expand=True, padx=8, pady=(8, 0))

    # We'll recreate the widget when switching files
    current_widget = {"w": None}

    def render_table_for(fid: int):
        # clear previous
        for w in table_frame.grid_slaves():
            w.destroy()
        if current_widget["w"] is not None:
            try:
                current_widget["w"].destroy()
            except Exception:
                pass
            current_widget["w"] = None

        df_res = analysis_results_per_file.get(fid, pd.DataFrame())

        if df_res is None or df_res.empty:
            note = tk.Label(table_frame, text="No valid results for this file.", font=("Arial", 11))
            note.grid(row=0, column=0, sticky="nw")
            current_widget["w"] = note
            return

        cols = list(df_res.columns)
        tree = ttk.Treeview(table_frame, columns=cols, show="headings")

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=max(120, min(260, int(9 * max(10, len(str(c)))))), anchor="w")

        for _, row in df_res.iterrows():
            tree.insert("", "end", values=[row[c] for c in cols])

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        current_widget["w"] = tree

    def on_select(event=None):
        sel = lst_files.curselection()
        if not sel:
            return
        fid = file_ids[sel[0]]
        render_table_for(fid)

    lst_files.bind("<<ListboxSelect>>", on_select)

    # Default selection
    if file_ids:
        lst_files.selection_set(0)
        lst_files.activate(0)
        render_table_for(file_ids[0])

    # Save all tables to a chosen folder
    def save_all_tables():
        target_dir = filedialog.askdirectory(title="Select folder to save all analysis tables")
        if not target_dir:
            return

        saved = 0
        for fid in file_ids:
            df_res = analysis_results_per_file.get(fid, pd.DataFrame())
            if df_res is None or df_res.empty:
                continue
            base = analysis_file_names.get(fid, os.path.basename(file_paths[fid]))
            stem, _ = os.path.splitext(base)
            out_path = os.path.join(target_dir, f"{stem}_analysis.csv")
            try:
                df_res.to_csv(out_path, index=False)
                saved += 1
            except Exception as e:
                print(f"[Save error] {out_path}: {e}")

        messagebox.showinfo("Export complete", f"Saved {saved} table(s) to:\n{target_dir}")

    btn_save_all = tk.Button(right, text="Save All Tables", command=save_all_tables)
    btn_save_all.pack(pady=10)


def show_plots():
    """Multi-file Plots: left file list; right plots for selected file; save 4 PNGs for that file to a chosen folder."""
    global analysis_results_per_file, analysis_file_names, file_paths

    plot_window = tk.Toplevel()
    set_default_window_size(plot_window, 0.92, 0.88, 1100, 720)
    plot_window.title("Analysis Plots")
    plot_window.geometry("1100x800")

    # ---------- Layout ----------
    root = tk.Frame(plot_window)
    root.pack(fill="both", expand=True)

    # Left panel: files list
    left = tk.Frame(root, width=240)
    left.pack(side="left", fill="y")
    tk.Label(left, text="Files", font=("Arial", 12, "bold")).pack(pady=(10, 6))
    lst_files = tk.Listbox(left, height=28)
    lst_files.pack(fill="y", expand=False, padx=10, pady=(0, 10))

    file_ids = list(range(len(file_paths)))
    for fid in file_ids:
        name = analysis_file_names.get(fid, os.path.basename(file_paths[fid]))
        lst_files.insert(tk.END, name)

    # Right panel: scrollable canvas for the figure + Save button
    right = tk.Frame(root)
    right.pack(side="right", fill="both", expand=True)

    canvas_frame = tk.Frame(right)
    canvas_frame.pack(fill=tk.BOTH, expand=True)

    canvas = Canvas(canvas_frame)
    v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
    h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Hold current figure/axes so Save button can access them
    current = {"fid": None, "fig": None, "axes": None, "tk_widget": None}

    def clear_current_canvas():
        if current["tk_widget"] is not None:
            try:
                current["tk_widget"].destroy()
            except Exception:
                pass
            current["tk_widget"] = None
        if current["fig"] is not None:
            try:
                plt.close(current["fig"])
            except Exception:
                pass
            current["fig"] = None
            current["axes"] = None

    def render_plots_for(fid: int):
        """Build the 2x2 plots for the selected file."""
        clear_current_canvas()

        df_res = analysis_results_per_file.get(fid, pd.DataFrame())
        if df_res is None or df_res.empty:
            msg = tk.Label(scrollable_frame, text="No plots available for this file.", font=("Arial", 11))
            msg.pack(pady=20)
            current["tk_widget"] = msg
            current["fid"] = fid
            return

        fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=150)

        # Plot Shape vs Threshold
        axes[0, 0].plot(df_res["Threshold"], df_res["Shape"], linestyle='-')
        axes[0, 0].set_ylabel("Shape (ξ)", fontsize=9); axes[0, 0].set_xlabel(""); axes[0, 0].grid()

        # Plot Scale vs Threshold
        axes[0, 1].plot(df_res["Threshold"], df_res["Scale"], linestyle='-')
        axes[0, 1].set_ylabel("Scale (σ)", fontsize=9); axes[0, 1].set_xlabel(""); axes[0, 1].grid()

        # Return values vs Threshold
        axes[1, 0].plot(df_res["Threshold"], df_res["Return value (GPD)"], linestyle='-', label="GPD")
        axes[1, 0].plot(df_res["Threshold"], df_res["Return value (ED)"], linestyle='-', label="ED")
        axes[1, 0].set_xlabel("Threshold", fontsize=9); axes[1, 0].set_ylabel("Return Value", fontsize=9)
        axes[1, 0].legend(fontsize=7); axes[1, 0].grid()

        # AIC vs Threshold
        axes[1, 1].plot(df_res["Threshold"], df_res["AIC_GPD"], linestyle='-', label='GPD')
        axes[1, 1].plot(df_res["Threshold"], df_res["AIC_ED"], linestyle='-', label='ED')
        axes[1, 1].set_xlabel("Threshold", fontsize=9); axes[1, 1].set_ylabel("AIC", fontsize=9)
        axes[1, 1].legend(fontsize=7); axes[1, 1].grid()

        for ax in axes.flatten():
            plt.setp(ax.get_yticklabels(), rotation=70, fontsize=7)
            plt.setp(ax.get_xticklabels(), fontsize=7)

        plt.subplots_adjust(left=0.08, right=0.98, top=0.99, bottom=0.10, hspace=0.25, wspace=0.22)

        canvas_fig = FigureCanvasTkAgg(fig, master=scrollable_frame)
        widget = canvas_fig.get_tk_widget()
        widget.pack()
        canvas_fig.draw()

        current["fid"] = fid
        current["fig"] = fig
        current["axes"] = axes
        current["tk_widget"] = widget

    def on_select(event=None):
        sel = lst_files.curselection()
        if not sel:
            return
        fid = file_ids[sel[0]]
        render_plots_for(fid)

    lst_files.bind("<<ListboxSelect>>", on_select)

    # Default file
    if file_ids:
        lst_files.selection_set(0)
        lst_files.activate(0)
        render_plots_for(file_ids[0])

    def save_plots_for_all():
        """Ask for a folder and save the 4 plots for EVERY file (one set per file)."""
        target_dir = filedialog.askdirectory(title="Select folder to save plots")
        if not target_dir:
            return
    
        saved_files = 0
    
        for fid in file_ids:
            df_res = analysis_results_per_file.get(fid, pd.DataFrame())
            if df_res is None or df_res.empty:
                continue
    
            # Build the four plots off-screen for this file
            fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=150)
    
            # Plot Shape vs Threshold
            axes[0, 0].plot(df_res["Threshold"], df_res["Shape"], linestyle='-')
            axes[0, 0].set_ylabel("Shape (ξ)", fontsize=9); axes[0, 0].grid()
    
            # Plot Scale vs Threshold
            axes[0, 1].plot(df_res["Threshold"], df_res["Scale"], linestyle='-')
            axes[0, 1].set_ylabel("Scale (σ)", fontsize=9); axes[0, 1].grid()
    
            # Return values vs Threshold
            axes[1, 0].plot(df_res["Threshold"], df_res["Return value (GPD)"], linestyle='-', label="GPD")
            axes[1, 0].plot(df_res["Threshold"], df_res["Return value (ED)"], linestyle='-', label="ED")
            axes[1, 0].set_xlabel("Threshold", fontsize=9); axes[1, 0].set_ylabel("Return Value", fontsize=9)
            axes[1, 0].legend(fontsize=7); axes[1, 0].grid()
    
            # AIC vs Threshold
            axes[1, 1].plot(df_res["Threshold"], df_res["AIC_GPD"], linestyle='-', label='GPD')
            axes[1, 1].plot(df_res["Threshold"], df_res["AIC_ED"], linestyle='-', label='ED')
            axes[1, 1].set_xlabel("Threshold", fontsize=9); axes[1, 1].set_ylabel("AIC", fontsize=9)
            axes[1, 1].legend(fontsize=7); axes[1, 1].grid()
    
            for ax in axes.flatten():
                plt.setp(ax.get_yticklabels(), rotation=70, fontsize=7)
                plt.setp(ax.get_xticklabels(), fontsize=7)
    
            # Save each subplot separately at 400 dpi with file-specific stem
            base = analysis_file_names.get(fid, os.path.basename(file_paths[fid]))
            stem, _ = os.path.splitext(base)
            names = [
                f"{stem}_shape_vs_threshold.png",
                f"{stem}_scale_vs_threshold.png",
                f"{stem}_return_values_vs_threshold.png",
                f"{stem}_aic_vs_threshold.png",
            ]
    
            for i, ax in enumerate(axes.flatten()):
                single_fig = plt.figure()
                new_ax = single_fig.add_subplot(111)
                for line in ax.lines:
                    new_ax.plot(line.get_xdata(), line.get_ydata(),
                                linestyle=line.get_linestyle(),
                                label=line.get_label() if line.get_label() != "_nolegend_" else None,
                                color=line.get_color())
                new_ax.set_xlabel("Threshold", fontsize=9)
                new_ax.set_ylabel(ax.get_ylabel(), fontsize=9)
                if ax.get_legend() is not None:
                    new_ax.legend(fontsize=7)
                new_ax.grid()
    
                out_path = os.path.join(target_dir, names[i])
                single_fig.savefig(out_path, dpi=400)
                plt.close(single_fig)
    
            plt.close(fig)
            saved_files += 1
    
        messagebox.showinfo("Success", f"Saved plots for {saved_files} file(s) in:\n{target_dir}")


    btn_save = tk.Button(right, text="Save Plots for all Files", command=save_plots_for_all)
    btn_save.pack(pady=10)

    plot_window.mainloop()

def embed_agreement_plots(frame_plots, df, threshold_col, shape_col, scale_col, XX, YY, stable_ranges_shape, stable_ranges_scale):
    """Embeds the three agreement plots inside the frame_plots frame, no scrolling."""

    # Ensure the frame_plots is ready and has calculated its width
    frame_plots.update_idletasks()
    width_inches = max(frame_plots.winfo_width() / frame_plots.winfo_fpixels('1i'), 6)

    # Create the figure with 3 subplots stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(width_inches, 10), dpi=100)

    # Plot 1: Shape Parameter Stability
    axes[0].plot(df[threshold_col], df[shape_col], linestyle='--', label="Shape")
    for (start, end) in stable_ranges_shape:
        axes[0].fill_between(df[threshold_col], df[shape_col],
                             where=(df[threshold_col] >= start) & (df[threshold_col] <= end),
                             alpha=0.3, label=f"Stable ({start:.2f} - {end:.2f})")
    axes[0].set_title("Shape Parameter Stability")
    #axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Shape")
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.35, 1.0), ncol=2)
    axes[0].grid()
    axes[0].tick_params(labelbottom=False)  # Hide x-axis ticks & labels
    
    # Plot 2: Scale Parameter Stability
    axes[1].plot(df[threshold_col], df[scale_col], linestyle='--', label="Scale")
    for (start, end) in stable_ranges_scale:
        axes[1].fill_between(df[threshold_col], df[scale_col],
                             where=(df[threshold_col] >= start) & (df[threshold_col] <= end),
                             alpha=0.3, label=f"Stable ({start:.2f} - {end:.2f})")
    axes[1].set_title("Scale Parameter Stability")
    #axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Scale")
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.35, 0.4), ncol=2)
    axes[1].grid()
    axes[1].tick_params(labelbottom=False)  # Hide x-axis ticks & labels

    # Plot 3: Return Value Agreement
    axes[2].plot(df[threshold_col], df['normalized_abs_diff'], label="Norm Abs Diff (0-1)", alpha=0.5)
    axes[2].plot(df[threshold_col], df['smoothed_normalized_diff'], label="Smoothed Norm Abs Diff", linewidth=2)
    if XX is not None:
        axes[2].axvline(x=XX, color='green', linestyle='--', label=f'Agreement XX@{XX:.2f}')
    if YY is not None:
        axes[2].axvline(x=YY, color='red', linestyle='--', label=f'Intersection YY@{YY:.2f}')
    axes[2].set_title("Return Value Agreement")
    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("Normalized Abs Diff")
    axes[2].legend(loc='upper center', bbox_to_anchor=(0.65, 1.00), ncol=2)
    axes[2].grid()

    # Adjust layout to prevent overlap
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves space for the x-axis label
    fig.subplots_adjust(bottom=0.10)  # Ensures the last x-axis label is not cut off


    # Embed the Matplotlib figure into frame_plots
    canvas = FigureCanvasTkAgg(fig, master=frame_plots)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()
    return fig

def open_next_step():
    """
    Multi-file Next Step window:
    - Left sidebar split: top = files list, bottom = threshold details (fixed size)
    - Right side: plots (unchanged)
    """
    # We will read per-file results and names from globals
    global analysis_results_per_file, analysis_file_names, file_paths
    
    
    threshold_col = "Threshold"
    gpd_col = "Return value (GPD)"
    ed_col = "Return value (ED)"
    shape_col = "Shape"
    scale_col = "Scale"
    
    # lightweight caches scoped to this window
    if not hasattr(open_next_step, "_finalT"): open_next_step._finalT = {}
    if not hasattr(open_next_step, "_figs"):   open_next_step._figs   = {}
    final_thresholds = open_next_step._finalT
    fig_cache        = open_next_step._figs
    
    next_window = tk.Tk()
    next_window.state('zoomed')
    next_window.title("Next Step")

    # Main container
    frame_main = tk.Frame(next_window)
    frame_main.pack(fill=tk.BOTH, expand=True)

    # LEFT SIDEBAR (fixed width)
    left_sidebar = tk.Frame(frame_main, width=360)   # ~ fixed sidebar width
    left_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 2))
    left_sidebar.pack_propagate(False)  # keep fixed width

    # Split left sidebar into: TOP (files list) + BOTTOM (details panel)
    files_panel = tk.Frame(left_sidebar, height=280)   # top half-ish for files list
    files_panel.pack(side=tk.TOP, fill=tk.X, pady=(6, 8))
    files_panel.pack_propagate(False)

    details_panel = tk.Frame(left_sidebar)             # bottom: your existing details (fixed size later)
    details_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # RIGHT plots area (expands)
    frame_plots = tk.Frame(frame_main)
    frame_plots.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 5), ipadx=8)

    # --- FILES LIST (top sub-panel) ---
    tk.Label(files_panel, text="Files", font=("Arial", 12, "bold"), anchor="w").pack(fill="x")

    files_list = tk.Listbox(files_panel, height=12, exportselection=False)
    files_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def on_pick(event=None):
        sel = files_list.curselection()
        if not sel:
            return
        fid = int(sel[0])
        render_for_file(fid)
    
    files_list.bind("<<ListboxSelect>>", on_pick)
    
    files_scroll = tk.Scrollbar(files_panel, orient=tk.VERTICAL, command=files_list.yview)
    files_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    files_list.config(yscrollcommand=files_scroll.set)

    # Populate file names
    for fid, p in enumerate(file_paths):
        name = analysis_file_names.get(fid, os.path.basename(p))
        files_list.insert(tk.END, name)

    
    # --- DETAILS PANEL HEADINGS (static) ---
    hdr_ranges = tk.Label(details_panel, text="Threshold ranges for stable parameters",
                          font=("Arial", 12, "bold"), anchor='w', justify="left")
    hdr_ranges.pack(pady=10, fill="x")
    
    lbl_shape_ranges = tk.Label(details_panel, text="Stable Ranges (Shape):\n—",
                                font=("Arial", 10), anchor="w", justify="left")
    lbl_shape_ranges.pack(pady=5, fill="x")
    
    lbl_scale_ranges = tk.Label(details_panel, text="Stable Ranges (Scale):\n—",
                                font=("Arial", 10), anchor="w", justify="left")
    lbl_scale_ranges.pack(pady=5, fill="x")
    
    hdr_agree = tk.Label(details_panel, text="Return Value Agreement from GPD and ED",
                         font=("Arial", 12, "bold"), anchor='w', justify="left")
    hdr_agree.pack(pady=10, fill="x")
    
    lbl_xx = tk.Label(details_panel, text="XX (First agreement threshold): —",
                      font=("Arial", 10), anchor='w', justify="left")
    lbl_xx.pack(pady=5, fill="x")
    
    lbl_yy = tk.Label(details_panel, text="YY (Crossover point): —",
                      font=("Arial", 10), anchor='w', justify="left")
    lbl_yy.pack(pady=5, fill="x")
    
    hdr_overlap = tk.Label(details_panel, text="Overlapping regions identification",
                           font=("Arial", 12, "bold"), anchor='w', justify="left")
    hdr_overlap.pack(pady=10, fill="x")
    
    lbl_overlap_exist = tk.Label(details_panel, text="Overlapping Stable Regions Exist: —",
                                 font=("Arial", 10), anchor='w', justify="left")
    lbl_overlap_exist.pack(pady=5, fill="x")
    
    lbl_overlap_ranges = tk.Label(details_panel, text="Overlapping Ranges:\n—",
                                  font=("Arial", 10), anchor="w", justify="left")
    lbl_overlap_ranges.pack(pady=5, fill="x")
    
    hdr_final = tk.Label(details_panel, text="Final Threshold Calculated.",
                         font=("Arial", 12, "bold"), anchor='w', justify="left")
    hdr_final.pack(pady=10, fill="x")
    
    # --- Final Threshold box (bold red) ---
    final_box = tk.Frame(details_panel, bg="#cc0000", padx=2, pady=2)  # red border
    final_box.pack(pady=10, fill="x")
    
    final_inner = tk.Frame(final_box, bg="#fff5f5")
    final_inner.pack(fill="x")
    
    lbl_final_T = tk.Label(
        final_inner,
        text="Final Threshold (T): —",
        font=("Arial", 12, "bold"),
        fg="#cc0000",
        bg="#fff5f5",
        anchor="w",
        padx=8,
        pady=6
    )
    lbl_final_T.pack(fill="x")
    
    # --- Save buttons (all files) ---
    btn_row = tk.Frame(details_panel)
    btn_row.pack(pady=8, fill="x")
    
    def save_csv_all():
        # ensure every file has T in cache by rendering once if missing
        cur = files_list.curselection()
        current_sel = cur[0] if cur else None
    
        for fid in range(len(file_paths)):
            if fid not in final_thresholds:
                files_list.selection_clear(0, tk.END)
                files_list.selection_set(fid)
                files_list.activate(fid)
                render_for_file(fid)  # this fills final_thresholds[fid] and fig_cache[fid]
    
        # pick save location
        from tkinter import filedialog, messagebox
        out_path = filedialog.asksaveasfilename(
            title="Save thresholds CSV",
            defaultextension=".csv",
            filetypes=[("CSV files","*.csv"), ("All files","*.*")]
        )
        if not out_path:
            # restore selection and return
            if current_sel is not None:
                files_list.selection_clear(0, tk.END)
                files_list.selection_set(current_sel)
                files_list.activate(current_sel)
                render_for_file(current_sel)
            return
    
        import csv, os
        rows = [("file_name", "final_threshold")]
        for fid, p in enumerate(file_paths):
            name = analysis_file_names.get(fid, os.path.basename(p))
            T = final_thresholds.get(fid, None)
            rows.append((name, None if T is None else float(T)))
        try:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerows(rows)
            messagebox.showinfo("Saved", f"Thresholds CSV written to:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV:\n{e}")
    
        # restore previous selection
        if current_sel is not None:
            files_list.selection_clear(0, tk.END)
            files_list.selection_set(current_sel)
            files_list.activate(current_sel)
            render_for_file(current_sel)
    
    def save_plots_all():
        # ensure a figure exists for every file by rendering once if needed
        cur = files_list.curselection()
        current_sel = cur[0] if cur else None
    
        for fid in range(len(file_paths)):
            if fid not in fig_cache:
                files_list.selection_clear(0, tk.END)
                files_list.selection_set(fid)
                files_list.activate(fid)
                render_for_file(fid)  # fills fig_cache[fid]
    
        from tkinter import filedialog, messagebox
        import os
        save_dir = filedialog.askdirectory(title="Select folder to save all plots")
        if not save_dir:
            if current_sel is not None:
                files_list.selection_clear(0, tk.END)
                files_list.selection_set(current_sel)
                files_list.activate(current_sel)
                render_for_file(current_sel)
            return
    
        for fid, p in enumerate(file_paths):
            name = analysis_file_names.get(fid, os.path.basename(p))
            safe = os.path.splitext(os.path.basename(name))[0]
            out_png = os.path.join(save_dir, f"{safe}_agreement.png")
            fig = fig_cache.get(fid)
            if fig is None:
                continue
            try:
                fig.savefig(out_png, dpi=400)
            except Exception:
                # try a fresh save after a draw, in case backend needs it
                fig.canvas.draw_idle()
                fig.savefig(out_png, dpi=400)
    
        messagebox.showinfo("Saved", f"All plots saved in:\n{save_dir}")
    
        # restore previous selection
        if current_sel is not None:
            files_list.selection_clear(0, tk.END)
            files_list.selection_set(current_sel)
            files_list.activate(current_sel)
            render_for_file(current_sel)
    
    tk.Button(btn_row, text="Save CSV (all files)",   command=save_csv_all).pack(side=tk.LEFT, padx=4)
    tk.Button(btn_row, text="Save Plots (all files)", command=save_plots_all).pack(side=tk.LEFT, padx=4)

    # --- Bottom button bar: Back / Exit ---
    btn_bar = tk.Frame(next_window)
    btn_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=8)

    btn_back = tk.Button(
        btn_bar,
        text="Back",
        command=lambda: [next_window.destroy(), open_results_window()]
    )
    btn_back.pack(side=tk.LEFT, padx=8)

    btn_exit = tk.Button(btn_bar, text="Exit", command=next_window.destroy)
    btn_exit.pack(side=tk.LEFT, padx=8)     

    # ---- helpers (added in 2B) ----
    def clear_plots():
        """Remove any existing plot widgets before drawing new ones."""
        for w in frame_plots.winfo_children():
            w.destroy()
    
    def format_ranges_multiline(ranges, items_per_line=2):
        """Formats the stable ranges into multiple lines, displaying only 'items_per_line' per row."""
        if not ranges:
            return "No stable ranges identified"
        formatted_lines = []
        for i in range(0, len(ranges), items_per_line):
            line = " || ".join([f"{round(float(x), 2)} - {round(float(y), 2)}" for (x, y) in ranges[i:i+items_per_line]])
            formatted_lines.append(line)
        return "\n".join(formatted_lines)
    # ---- end helpers ----
    
    def render_for_file(fid: int):
        """
        Render details + plots for the selected file id.
        Uses analysis_results_per_file[fid] (DataFrame with your columns).
        """
        if fid not in analysis_results_per_file:
            clear_plots()
            lbl_shape_ranges.config(text="Stable Ranges (Shape):\n—")
            lbl_scale_ranges.config(text="Stable Ranges (Scale):\n—")
            lbl_xx.config(text="XX (First agreement threshold): —")
            lbl_yy.config(text="YY (Crossover point): —")
            lbl_overlap_exist.config(text="Overlapping Stable Regions Exist: —")
            lbl_overlap_ranges.config(text="Overlapping Ranges:\n—")
            lbl_final_T.config(text="Final Threshold (T): —")
            return
    
        # 1) Get this file's results
        df = analysis_results_per_file[fid].copy()
    
        # 2) Re-run your unchanged methodology on THIS df
        stable_ranges_shape = []
        stable_ranges_scale = []
    
        def normalize_values(values):
            return (values - np.mean(values)) / np.std(values)
    
        segment_size = max(5, min(10, int(len(df) * 0.1)))
    
        def compute_segment_slopes(thresholds, values, segment_size):
            slopes = []
            for i in range(0, len(thresholds) - segment_size + 1):
                x_segment = thresholds[i: i + segment_size].reshape(-1, 1)
                y_segment = values[i: i + segment_size]
                model = LinearRegression()
                model.fit(x_segment, y_segment)
                slope = abs(model.coef_[0])
                slopes.append((i, slope))
            return slopes
    
        def merge_stable_regions(stable_ranges):
            if not stable_ranges:
                return []
            stable_ranges.sort()
            avg_segment_size = np.mean([max_r - min_r for min_r, max_r in stable_ranges])
            merge_distance = avg_segment_size / 2
            merged = [stable_ranges[0]]
            for i in range(1, len(stable_ranges)):
                prev_min, prev_max = merged[-1]
                curr_min, curr_max = stable_ranges[i]
                if curr_min - prev_max <= merge_distance:
                    merged[-1] = (prev_min, max(prev_max, curr_max))
                else:
                    merged.append((curr_min, curr_max))
            return merged
    
        thresholds_np = df[threshold_col].values
        for param_col, bucket in [(shape_col, stable_ranges_shape), (scale_col, stable_ranges_scale)]:
            values = normalize_values(df[param_col].values)
            slopes = compute_segment_slopes(thresholds_np, values, segment_size)
            slopes.sort(key=lambda x: x[1])
            top_segments = slopes[:3]
            bucket.extend([(thresholds_np[s[0]], thresholds_np[s[0] + segment_size - 1]) for s in top_segments])
            bucket[:] = merge_stable_regions(bucket)
    
        # Agreement (unchanged)
        df['abs_diff'] = abs(df[gpd_col] - df[ed_col])
        min_abs = df['abs_diff'].min(); max_abs = df['abs_diff'].max()
        df['normalized_abs_diff'] = (df['abs_diff'] - min_abs) / (max_abs - min_abs) if max_abs > min_abs else 0.0
        df['smoothed_normalized_diff'] = df['normalized_abs_diff'].rolling(window=5, min_periods=1).mean()
    
        # XX
        agreement_indices = np.where(df['smoothed_normalized_diff'] < 0.1)[0]
        XX = df[threshold_col].iloc[agreement_indices[0]] if len(agreement_indices) > 0 else None
    
        # YY
        YY = None
        for i in range(len(df) - 1):
            if np.sign(df[gpd_col].iloc[i] - df[ed_col].iloc[i]) != np.sign(df[gpd_col].iloc[i+1] - df[ed_col].iloc[i+1]):
                t1, t2 = df[threshold_col].iloc[i], df[threshold_col].iloc[i+1]
                gpd1, gpd2 = df[gpd_col].iloc[i], df[gpd_col].iloc[i+1]
                ed1, ed2 = df[ed_col].iloc[i], df[ed_col].iloc[i+1]
                YY = t1 + (t2 - t1) * abs(gpd1 - ed1) / (abs(gpd1 - ed1) + abs(gpd2 - ed2))
                break
    
        def find_shape_scale_overlaps(shape_ranges, scale_ranges):
            overlaps = []
            for (s_min, s_max) in shape_ranges:
                for (t_min, t_max) in scale_ranges:
                    omin = max(s_min, t_min)
                    omax = min(s_max, t_max)
                    if omin < omax:
                        overlaps.append((omin, omax))
            return overlaps
    
        def center_of_region(rng):
            return (rng[0] + rng[1]) / 2
    
        def nearest_stable_region_center(value, shape_ranges, scale_ranges):
            all_ranges = [(r, 'shape') for r in shape_ranges] + [(r, 'scale') for r in scale_ranges]
            if not all_ranges:
                return None
            best_center = None; best_dist = float('inf')
            for (r, _) in all_ranges:
                c = center_of_region(r)
                dist = abs(c - value)
                if dist < best_dist:
                    best_dist = dist; best_center = c
            return (best_center, None) if best_center is not None else None
    
        def compute_T_case1(xx, yy, shape_ranges, scale_ranges):
            if overlap_exists:
                for (omin, omax) in all_overlaps:
                    overlap_center = center_of_region((omin, omax))
                    if xx < omin:
                        if yy <= omax:
                            return (overlap_center + yy) / 2
                        else:
                            return overlap_center
                    elif omin <= xx <= omax:
                        return (overlap_center + xx) / 2
                    else:
                        return (overlap_center + xx) / 2
                fallback_center = center_of_region(all_overlaps[-1])
                return (fallback_center + xx) / 2
            else:
                start, end = sorted([xx, yy])
                def fully_in_interval(ranges, low, high):
                    return [(rmin, rmax) for (rmin, rmax) in ranges if rmin >= low and rmax <= high]
                shape_in_int = fully_in_interval(shape_ranges, start, end)
                scale_in_int = fully_in_interval(scale_ranges, start, end)
                if shape_in_int or scale_in_int:
                    all_in_interval = shape_in_int + scale_in_int
                    best_center = min(all_in_interval, key=lambda r: abs(center_of_region(r) - xx), default=None)
                    return center_of_region(best_center) if best_center else xx
                else:
                    near_center = nearest_stable_region_center(xx, shape_ranges, scale_ranges)
                    return (xx + near_center[0]) / 2 if near_center else xx
    
        def compute_T_case2(xx, shape_ranges, scale_ranges):
            if overlap_exists:
                for (omin, omax) in all_overlaps:
                    overlap_center = center_of_region((omin, omax))
                    return (overlap_center + xx) / 2
                fallback_center = center_of_region(all_overlaps[-1])
                return (fallback_center + xx) / 2
            else:
                def stable_region_centers_after(xx, shape_ranges, scale_ranges):
                    return [center_of_region((rmin, rmax)) for (rmin, rmax) in shape_ranges + scale_ranges if center_of_region((rmin, rmax)) >= xx]
                after_centers = stable_region_centers_after(xx, shape_ranges, scale_ranges)
                if after_centers:
                    return min(after_centers, key=lambda c: abs(c - xx))
                else:
                    near_center = nearest_stable_region_center(xx, shape_ranges, scale_ranges)
                    return (xx + near_center[0]) / 2 if near_center else xx
    
        all_overlaps = find_shape_scale_overlaps(stable_ranges_shape, stable_ranges_scale)
        overlap_exists = len(all_overlaps) > 0
    
        T = None
        if XX is not None:
            if YY is not None:
                T = compute_T_case1(XX, YY, stable_ranges_shape, stable_ranges_scale)
            else:
                T = compute_T_case2(XX, stable_ranges_shape, stable_ranges_scale)
    
        # 3) Update details panel labels
        lbl_shape_ranges.config(text=f"Stable Ranges (Shape):\n{format_ranges_multiline(stable_ranges_shape)}")
        lbl_scale_ranges.config(text=f"Stable Ranges (Scale):\n{format_ranges_multiline(stable_ranges_scale)}")
        lbl_xx.config(text=f"XX (First agreement threshold): {XX if XX is not None else '—'}")
        lbl_yy.config(text=f"YY (Crossover point): {YY if YY is not None else '—'}")
        lbl_overlap_exist.config(text=f"Overlapping Stable Regions Exist: {overlap_exists}")
        lbl_overlap_ranges.config(text=f"Overlapping Ranges:\n{format_ranges_multiline(all_overlaps)}" if overlap_exists else "Overlapping Ranges:\n—")
        lbl_final_T.config(text=f"Final Threshold (T): {T if T is not None else '—'}")
    
        # 4) Re-draw plots on the right
        clear_plots()
        fig_drawn = embed_agreement_plots(frame_plots, df, threshold_col, shape_col, scale_col, XX, YY, stable_ranges_shape, stable_ranges_scale)

        # cache for saving later (no recomputation)
        final_thresholds[fid] = T
        fig_cache[fid] = fig_drawn

    # --- Select first file by default (after list is populated and render_for_file exists) ---
    if file_paths and files_list.size() > 0:
        files_list.selection_set(0)
        files_list.activate(0)
        render_for_file(0)

# ------------------ Start the Application ------------------
open_main_window()
