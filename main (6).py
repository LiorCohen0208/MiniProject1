import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Base Neuron Classes
class GeneralNeuron:
    def __init__(self, firing_rate: float):
        self.firing_rate = firing_rate

    def activate(self, stimulus_strength: float):
        pass  # Define activation logic

class SensoryNeuron(GeneralNeuron):
    def __init__(self, firing_rate: float, receptor_type: str):
        super().__init__(firing_rate)
        self.receptor_type = receptor_type

    def sense_stimulus(self, stimulus):
        pass  # Define sensing logic

class MotorNeuron(GeneralNeuron):
    def __init__(self, firing_rate: float, target_muscle: str):
        super().__init__(firing_rate)
        self.target_muscle = target_muscle

    def control_muscle(self, activation_level: float):
        pass  # Define control logic

# Specific Neuron Types
class Photoreceptor(SensoryNeuron):
    def __init__(self, firing_rate: float):
        super().__init__(firing_rate, receptor_type="light")

    def light_detection(self, light_intensity: float):
        pass

class Mechanoreceptor(SensoryNeuron):
    def __init__(self, firing_rate: float):
        super().__init__(firing_rate, receptor_type="pressure")

    def pressure_detection(self, pressure_level: float):
        pass  # Define pressure-specific behavior

class AlphaMotorNeuron(MotorNeuron):
    def __init__(self, firing_rate: float):
        super().__init__(firing_rate, target_muscle="skeletal muscle")

    def skeletal_muscle_control(self, activation_level: float):
        pass  # Define skeletal muscle behavior

class GammaMotorNeuron(MotorNeuron):
    def __init__(self, firing_rate: float):
        super().__init__(firing_rate, target_muscle="muscle spindle")

    def muscle_spindle_control(self, activation_level: float):
        pass  # Define muscle spindle behavior

# Signal Processing Functions
def downsample_signal(signal, step=5):
    diff_signal = np.diff(signal)
    extrema_indices = np.where(np.diff(np.sign(diff_signal)) != 0)[0] + 1
    downsampled_indices = np.unique(np.concatenate((np.arange(0, len(signal), step), extrema_indices)))
    return signal[downsampled_indices], downsampled_indices

# Laptop Data Analysis Functions
def load_laptop_data(file_path):
    """Load laptop dataset from CSV file."""
    return pd.read_csv(file_path)

def plot_laptop_prices(df):
    """Plot laptop prices."""
    plt.figure(figsize=(10, 6))
    plt.plot(df["Price (Euro)"], marker='o', linestyle='-', color='b')
    plt.title("Laptop Prices")
    plt.xlabel("Laptop Index")
    plt.ylabel("Price (Euro)")
    plt.grid(True)
    plt.show()

def average_price_per_company(df):
    """Calculate and print average laptop prices by company."""
    average_prices = df.groupby("Company")["Price (Euro)"].mean()
    most_expensive_company = average_prices.idxmax()
    highest_average_price = average_prices.max()

    print("Average laptop price for each company:")
    print(average_prices)
    print(f"\nThe company with the most expensive laptops on average is {most_expensive_company}, "
          f"with an average price of {highest_average_price:.2f} Euros.")

def standardize_operating_systems(df):
    """Standardize operating system names."""
    print("Original Operating Systems:")
    print(df["OpSys"].unique())

    df.loc[df["OpSys"].isin(["Windows 10", "Windows 7"]), "OpSys"] = "Windows"
    df.loc[df["OpSys"].isin(["mac OS X", "MacOS X"]), "OpSys"] = "macOS"

    print("\nStandardized Operating Systems:")
    print(df["OpSys"].unique())
    return df

def plot_price_distribution_by_os(df):
    """Plot price distribution for each operating system."""
    unique_os = df["OpSys"].unique()

    for os in unique_os:
        plt.figure(figsize=(8, 6))
        os_data = df[df["OpSys"] == os]["Price (Euro)"]
        plt.hist(os_data, bins=10, color='skyblue', edgecolor='black')
        plt.title(f"Price Distribution for {os}")
        plt.xlabel("Price (Euro)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

def plot_ram_vs_price(df):
    """Plot RAM vs laptop price."""
    plt.figure(figsize=(8, 6))
    plt.scatter(df["RAM (GB)"], df["Price (Euro)"], color='orange', alpha=0.5)
    plt.title("Relationship Between RAM and Laptop Price")
    plt.xlabel("RAM (GB)")
    plt.ylabel("Price (Euro)")
    plt.grid(True)
    plt.show()

def create_storage_type_column(df):
    """Create a column for storage type."""
    df["Storage type"] = df["Memory"].str.split().str[-1]
    print(df[["Memory", "Storage type"]])
    return df

def plot_screen_size_vs_price(df):
    """Plot screen size vs laptop price."""
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Inches"], df["Price (Euro)"], color='green', alpha=0.6)
    plt.title("Relationship Between Screen Size and Laptop Price")
    plt.xlabel("Screen Size (Inches)")
    plt.ylabel("Price (Euro)")
    plt.grid(True)
    plt.show()

def plot_processor_vs_price(df):
    """Plot price distribution by processor type."""
    plt.figure(figsize=(10, 6))
    df.boxplot(column="Price (Euro)", by="Processor", grid=False, patch_artist=True, 
               medianprops=dict(color='red', linewidth=2))
    plt.title("Price Distribution by Processor Type")
    plt.suptitle("")
    plt.xlabel("Processor Type")
    plt.ylabel("Price (Euro)")
    plt.grid(True)
    plt.show()

def main():
    # Signal Downsampling Example
    x = np.linspace(0, 1, 100)  # 100 points between 0 and 1
    signal = np.sin(2 * np.pi * 5 * x)

    downsampled_signal, downsampled_indices = downsample_signal(signal, step=5)

    plt.plot(x, signal, label="Original Signal")
    plt.scatter(x[downsampled_indices], downsampled_signal, color='red', label="Downsampled Points")
    plt.title("Simplified Signal Downsampling")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Original Length: {len(signal)}")
    print(f"Downsampled Length: {len(downsampled_signal)}")
    print(f"Reduction Ratio: {len(downsampled_signal) / len(signal):.2f}")

    # Laptop Data Analysis
    file_path = "laptop_price - dataset.csv"  
    df = load_laptop_data(file_path)

    plot_laptop_prices(df)
    average_price_per_company(df)
    # df = standardize_operating_systems(df)
    # plot_price_distribution_by_os(df)
    # plot_ram_vs_price(df)
    # create_storage_type_column(df)
    # plot_screen_size_vs_price(df)
    # plot_processor_vs_price(df)

if __name__ == "__main__":
    main()


# בונוס 
# מה הקשר בין גודל המסך למחיר המחשב הנייד?

# אנליזה: ניתוח כיצד גודל המסך משפיע על המחיר.
# ויזואליזציה: גרף פיזור שמראה את הקשר בין גודל המסך למחיר.
# כיצד משפיע סוג המעבד על המחיר של מחשבים ניידים?

# אנליזה: ניתוח כיצד סוג המעבד משפיע על המחיר.
# ויזואליזציה: גרף Box plot שמראה את הפיזור של המחירים לפי סוג המעבד.
# מהו הממוצע של המחירים לפי גודל הזיכרון RAM?

# אנליזה: חישוב ממוצע המחירים לפי גודל הזיכרון RAM.
# ויזואליזציה: גרף עמודות שמראה את ממוצע המחירים לפי גודל הזיכרון.
