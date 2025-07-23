import time
import sys
import math
import heapq
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import psutil

class SortingAlgorithms:
    @staticmethod
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr

    @staticmethod
    def insertion_sort(arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr

    @staticmethod
    def selection_sort(arr):
        for i in range(len(arr)):
            min_idx = i
            for j in range(i+1, len(arr)):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr

    @staticmethod
    def merge_sort(arr):
        if len(arr) > 1:
            mid = len(arr) // 2
            left = arr[:mid]
            right = arr[mid:]
            
            SortingAlgorithms.merge_sort(left)
            SortingAlgorithms.merge_sort(right)
            
            i = j = k = 0
            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                k += 1
            
            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1
            
            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1
        return arr

    @staticmethod
    def quick_sort(arr):
        def partition(arr, low, high):
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            return i + 1
        
        def quick_sort_helper(arr, low, high):
            if low < high:
                pi = partition(arr, low, high)
                quick_sort_helper(arr, low, pi - 1)
                quick_sort_helper(arr, pi + 1, high)
        
        if len(arr) > 1:
            quick_sort_helper(arr, 0, len(arr) - 1)
        return arr

    @staticmethod
    def heap_sort(arr):
        def heapify(arr, n, i):
            largest = i
            l = 2 * i + 1
            r = 2 * i + 2
            
            if l < n and arr[l] > arr[largest]:
                largest = l
            if r < n and arr[r] > arr[largest]:
                largest = r
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)
        
        n = len(arr)
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            heapify(arr, i, 0)
        return arr

    @staticmethod
    def counting_sort(arr):
        if not arr:
            return arr
        
        # Handle floats by scaling
        is_float = any(isinstance(x, float) for x in arr)
        if is_float:
            scale = 1000
            scaled_arr = [int(x * scale) for x in arr]
            min_val = min(scaled_arr)
            max_val = max(scaled_arr)
        else:
            min_val = min(arr)
            max_val = max(arr)
            scaled_arr = arr
        
        count = [0] * (max_val - min_val + 1)
        
        for num in scaled_arr:
            count[num - min_val] += 1
        
        result = []
        for i, cnt in enumerate(count):
            if is_float:
                result.extend([(i + min_val) / scale] * cnt)
            else:
                result.extend([i + min_val] * cnt)
        
        return result

    @staticmethod
    def radix_sort(arr):
        if not arr:
            return arr
        
        # Handle floats and negatives
        is_float = any(isinstance(x, float) for x in arr)
        if is_float:
            scale = 1000
            scaled_arr = [int(abs(x) * scale) for x in arr]
        else:
            scaled_arr = [abs(x) for x in arr]
        
        # Separate positives and negatives
        positives = []
        negatives = []
        
        for i, x in enumerate(arr):
            if x >= 0:
                positives.append((scaled_arr[i], x))
            else:
                negatives.append((scaled_arr[i], x))
        
        def counting_sort_for_radix(arr, exp):
            output = [0] * len(arr)
            count = [0] * 10
            
            for item in arr:
                index = (item[0] // exp) % 10
                count[index] += 1
            
            for i in range(1, 10):
                count[i] += count[i - 1]
            
            i = len(arr) - 1
            while i >= 0:
                index = (arr[i][0] // exp) % 10
                output[count[index] - 1] = arr[i]
                count[index] -= 1
                i -= 1
            
            return output
        
        # Sort positives
        if positives:
            max_val = max(item[0] for item in positives)
            exp = 1
            while max_val // exp > 0:
                positives = counting_sort_for_radix(positives, exp)
                exp *= 10
        
        # Sort negatives
        if negatives:
            max_val = max(item[0] for item in negatives)
            exp = 1
            while max_val // exp > 0:
                negatives = counting_sort_for_radix(negatives, exp)
                exp *= 10
            negatives.reverse()  # Reverse for descending order of negatives
        
        result = [item[1] for item in negatives] + [item[1] for item in positives]
        return result

    @staticmethod
    def shell_sort(arr):
        n = len(arr)
        gap = n // 2
        
        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                while j >= gap and arr[j - gap] > temp:
                    arr[j] = arr[j - gap]
                    j -= gap
                arr[j] = temp
            gap //= 2
        return arr

    @staticmethod
    def bucket_sort(arr):
        if not arr:
            return arr
        
        # Determine range
        min_val, max_val = min(arr), max(arr)
        bucket_count = 10
        
        if min_val == max_val:
            return arr
            
        bucket_range = (max_val - min_val) / bucket_count
        
        buckets = [[] for _ in range(bucket_count)]
        
        for num in arr:
            if num == max_val:
                buckets[bucket_count - 1].append(num)
            else:
                bucket_index = int((num - min_val) / bucket_range)
                buckets[bucket_index].append(num)
        
        result = []
        for bucket in buckets:
            if bucket:
                bucket.sort()  # Use built-in sort for each bucket
                result.extend(bucket)
        
        return result

    @staticmethod
    def tim_sort(arr):
        return sorted(arr)  # Python's built-in Timsort

    @staticmethod
    def bitonic_sort(arr):
        def bitonic_merge(arr, up):
            if len(arr) > 1:
                k = len(arr) // 2
                for i in range(k):
                    if (arr[i] > arr[i + k]) == up:
                        arr[i], arr[i + k] = arr[i + k], arr[i]
                bitonic_merge(arr[:k], up)
                bitonic_merge(arr[k:], up)
        
        def bitonic_sort_rec(arr, up):
            if len(arr) > 1:
                k = len(arr) // 2
                bitonic_sort_rec(arr[:k], True)
                bitonic_sort_rec(arr[k:], False)
                bitonic_merge(arr, up)
        
        # Pad to power of 2 if necessary
        n = len(arr)
        power_of_2 = 1
        while power_of_2 < n:
            power_of_2 *= 2
        
        padded_arr = arr + [float('inf')] * (power_of_2 - n)
        bitonic_sort_rec(padded_arr, True)
        return padded_arr[:n]

    @staticmethod
    def tree_sort(arr):
        class TreeNode:
            def __init__(self, val):
                self.val = val
                self.left = None
                self.right = None
        
        def insert(root, val):
            if not root:
                return TreeNode(val)
            if val < root.val:
                root.left = insert(root.left, val)
            else:
                root.right = insert(root.right, val)
            return root
        
        def inorder(root, result):
            if root:
                inorder(root.left, result)
                result.append(root.val)
                inorder(root.right, result)
        
        if not arr:
            return arr
        
        root = None
        for val in arr:
            root = insert(root, val)
        
        result = []
        inorder(root, result)
        return result

    @staticmethod
    def cycle_sort(arr):
        writes = 0
        
        for cycle_start in range(len(arr) - 1):
            item = arr[cycle_start]
            pos = cycle_start
            
            for i in range(cycle_start + 1, len(arr)):
                if arr[i] < item:
                    pos += 1
            
            if pos == cycle_start:
                continue
            
            while item == arr[pos]:
                pos += 1
            
            arr[pos], item = item, arr[pos]
            writes += 1
            
            while pos != cycle_start:
                pos = cycle_start
                for i in range(cycle_start + 1, len(arr)):
                    if arr[i] < item:
                        pos += 1
                
                while item == arr[pos]:
                    pos += 1
                
                arr[pos], item = item, arr[pos]
                writes += 1
        
        return arr

    @staticmethod
    def strand_sort(arr):
        def merge_lists(list1, list2):
            result = []
            i = j = 0
            while i < len(list1) and j < len(list2):
                if list1[i] <= list2[j]:
                    result.append(list1[i])
                    i += 1
                else:
                    result.append(list2[j])
                    j += 1
            result.extend(list1[i:])
            result.extend(list2[j:])
            return result
        
        if not arr:
            return arr
        
        arr = arr.copy()
        result = []
        
        while arr:
            strand = [arr.pop(0)]
            i = 0
            while i < len(arr):
                if arr[i] >= strand[-1]:
                    strand.append(arr.pop(i))
                else:
                    i += 1
            result = merge_lists(result, strand)
        
        return result

    @staticmethod
    def cocktail_shaker_sort(arr):
        n = len(arr)
        swapped = True
        start = 0
        end = n - 1
        
        while swapped:
            swapped = False
            
            for i in range(start, end):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
            
            if not swapped:
                break
            
            end -= 1
            swapped = False
            
            for i in range(end - 1, start - 1, -1):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
            
            start += 1
        
        return arr

    @staticmethod
    def comb_sort(arr):
        gap = len(arr)
        shrink = 1.3
        sorted_flag = False
        
        while not sorted_flag:
            gap = int(gap / shrink)
            if gap <= 1:
                gap = 1
                sorted_flag = True
            
            i = 0
            while i + gap < len(arr):
                if arr[i] > arr[i + gap]:
                    arr[i], arr[i + gap] = arr[i + gap], arr[i]
                    sorted_flag = False
                i += 1
        
        return arr

    @staticmethod
    def gnome_sort(arr):
        index = 0
        while index < len(arr):
            if index == 0 or arr[index] >= arr[index - 1]:
                index += 1
            else:
                arr[index], arr[index - 1] = arr[index - 1], arr[index]
                index -= 1
        return arr

    @staticmethod
    def odd_even_sort(arr):
        n = len(arr)
        sorted_flag = False
        
        while not sorted_flag:
            sorted_flag = True
            
            # Odd indexed elements
            for i in range(1, n - 1, 2):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    sorted_flag = False
            
            # Even indexed elements
            for i in range(0, n - 1, 2):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    sorted_flag = False
        
        return arr

class SortingBenchmark:
    def __init__(self):
        self.algorithms = {
            'Bubble Sort': SortingAlgorithms.bubble_sort,
            'Insertion Sort': SortingAlgorithms.insertion_sort,
            'Selection Sort': SortingAlgorithms.selection_sort,
            'Merge Sort': SortingAlgorithms.merge_sort,
            'Quick Sort': SortingAlgorithms.quick_sort,
            'Heap Sort': SortingAlgorithms.heap_sort,
            'Counting Sort': SortingAlgorithms.counting_sort,
            'Radix Sort': SortingAlgorithms.radix_sort,
            'Shell Sort': SortingAlgorithms.shell_sort,
            'Bucket Sort': SortingAlgorithms.bucket_sort,
            'Tim Sort': SortingAlgorithms.tim_sort,
            'Bitonic Sort': SortingAlgorithms.bitonic_sort,
            'Tree Sort': SortingAlgorithms.tree_sort,
            'Cycle Sort': SortingAlgorithms.cycle_sort,
            'Strand Sort': SortingAlgorithms.strand_sort,
            'Cocktail Shaker Sort': SortingAlgorithms.cocktail_shaker_sort,
            'Comb Sort': SortingAlgorithms.comb_sort,
            'Gnome Sort': SortingAlgorithms.gnome_sort,
            'Odd-Even Sort': SortingAlgorithms.odd_even_sort
        }
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    
    def read_from_file(self, filename):
        """Read data from file"""
        try:
            with open(filename, 'r') as file:
                data = []
                for line in file:
                    line = line.strip()
                    if line:
                        # Try to parse as float first, then int
                        try:
                            if '.' in line:
                                data.append(float(line))
                            else:
                                data.append(int(line))
                        except ValueError:
                            # Skip invalid entries
                            continue
                return data
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    def write_results_to_file(self, filename, results):
        """Write sorting results to file"""
        try:
            with open(filename, 'w') as file:
                file.write("Sorting Algorithm Results\n")
                file.write("=" * 50 + "\n\n")
                
                for algo_name, result_data in results.items():
                    file.write(f"{algo_name}:\n")
                    file.write(f"Time: {result_data['time']:.6f} seconds\n")
                    file.write(f"Space: {result_data['space']:.6f} MB\n")
                    file.write("Sorted Array: ")
                    
                    # Write first 100 elements and last 100 elements if array is large
                    sorted_arr = result_data['sorted_array']
                    if len(sorted_arr) > 200:
                        file.write(str(sorted_arr[:100])[:-1] + " ... " + str(sorted_arr[-100:])[1:])
                    else:
                        file.write(str(sorted_arr))
                    
                    file.write("\n" + "-" * 30 + "\n\n")
            
            print(f"Results written to '{filename}'")
        except Exception as e:
            print(f"Error writing to file: {e}")
    
    def generate_graph(self, results):
        """Generate performance comparison graph"""
        try:
            algorithms = list(results.keys())
            times = [results[algo]['time'] for algo in algorithms]
            spaces = [results[algo]['space'] for algo in algorithms]
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Time comparison
            bars1 = ax1.bar(algorithms, times, color='skyblue', alpha=0.7)
            ax1.set_xlabel('Sorting Algorithms')
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('Time Comparison')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, time_val in zip(bars1, times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{time_val:.4f}s', ha='center', va='bottom', fontsize=8)
            
            # Space comparison
            bars2 = ax2.bar(algorithms, spaces, color='lightcoral', alpha=0.7)
            ax2.set_xlabel('Sorting Algorithms')
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.set_title('Memory Usage Comparison')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, space_val in zip(bars2, spaces):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{space_val:.4f}MB', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig('sorting_benchmark_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Combined chart
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # Plot time
            color = 'tab:blue'
            ax1.set_xlabel('Sorting Algorithms')
            ax1.set_ylabel('Time (seconds)', color=color)
            bars = ax1.bar(algorithms, times, color=color, alpha=0.6, label='Time')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot space
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Memory Usage (MB)', color=color)
            line = ax2.plot(algorithms, spaces, color=color, marker='o', linewidth=2, label='Memory')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Add legends
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            plt.title('Sorting Algorithm Performance Comparison')
            plt.tight_layout()
            plt.savefig('sorting_combined_chart.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Graphs saved as 'sorting_benchmark_results.png' and 'sorting_combined_chart.png'")
            
        except Exception as e:
            print(f"Error generating graph: {e}")
    
    def find_best_algorithms(self, results):
        """Find best algorithms by time and space"""
        if not results:
            return None, None
        
        best_time = min(results.items(), key=lambda x: x[1]['time'])
        best_space = min(results.items(), key=lambda x: x[1]['space'])
        
        return best_time, best_space
    
    def manual_input(self):
        """Get data from manual input"""
        print("Enter numbers separated by spaces (can be integers or floats):")
        try:
            input_str = input().strip()
            data = []
            for item in input_str.split():
                try:
                    if '.' in item:
                        data.append(float(item))
                    else:
                        data.append(int(item))
                except ValueError:
                    print(f"Skipping invalid input: {item}")
            return data
        except Exception as e:
            print(f"Error in manual input: {e}")
            return None
    
    def run_benchmark(self, data, is_file_data=False, output_file=None):
        """Run benchmark on all sorting algorithms"""
        if not data:
            print("No data to sort!")
            return
        
        print(f"\nSorting {len(data)} elements...")
        print(f"Data type: {'Mixed' if any(isinstance(x, float) for x in data) else 'Integer'}")
        
        initial_memory = self.get_memory_usage()
        print(f"Initial memory usage: {initial_memory:.6f} MB")
        
        results = {}
        
        # Skip slow algorithms for large datasets
        skip_slow = len(data) > 10000
        slow_algorithms = ['Bubble Sort', 'Selection Sort', 'Cycle Sort', 
                          'Strand Sort', 'Gnome Sort', 'Odd-Even Sort']
        
        print("\nRunning algorithms...")
        print("-" * 60)
        
        for algo_name, algo_func in self.algorithms.items():
            if skip_slow and algo_name in slow_algorithms:
                print(f"Skipping {algo_name} (too slow for large dataset)")
                continue
            
            try:
                # Create a copy of data for each algorithm
                data_copy = data.copy()
                
                # Measure memory before sorting
                mem_before = self.get_memory_usage()
                
                # Measure time
                start_time = time.perf_counter()
                sorted_data = algo_func(data_copy)
                end_time = time.perf_counter()
                
                # Measure memory after sorting
                mem_after = self.get_memory_usage()
                
                execution_time = end_time - start_time
                memory_usage = max(mem_after - mem_before, 0.000001)  # Avoid zero or negative
                
                results[algo_name] = {
                    'time': execution_time,
                    'space': memory_usage,
                    'sorted_array': sorted_data
                }
                
                if is_file_data:
                    print(f"{algo_name:<20}: {execution_time:.6f}s | {memory_usage:.6f} MB")
                else:
                    print(f"{algo_name:<20}: {execution_time:.6f}s | {memory_usage:.6f} MB")
                    print(f"Result: {sorted_data}")
                    print()
            
            except Exception as e:
                print(f"Error in {algo_name}: {e}")
                continue
        
        # Find best algorithms
        best_time, best_space = self.find_best_algorithms(results)
        
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS")
        print("="*60)
        
        if best_time:
            print(f"🏆 FASTEST ALGORITHM: {best_time[0]}")
            print(f"   Time: {best_time[1]['time']:.6f} seconds")
        
        if best_space:
            print(f"💾 MOST MEMORY EFFICIENT: {best_space[0]}")
            print(f"   Memory: {best_space[1]['space']:.6f} MB")
        
        # Generate graphs
        if results:
            self.generate_graph(results)
        
        # Write results to file if it's file data
        if is_file_data and output_file:
            self.write_results_to_file(output_file, results)
        
        return results
    
    def main(self):
        """Main function to run the program"""
        print("🔄 Sorting Algorithms Benchmark")
        print("=" * 40)
        print("1. Manual input")
        print("2. File input")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '1':
            data = self.manual_input()
            if data:
                self.run_benchmark(data, is_file_data=False)
        
        elif choice == '2':
            filename = input("Enter the input file name: ").strip()
            data = self.read_from_file(filename)
            
            if data:
                output_file = input("Enter output file name (press Enter for 'sorting_results.txt'): ").strip()
                if not output_file:
                    output_file = 'sorting_results.txt'
                
                self.run_benchmark(data, is_file_data=True, output_file=output_file)
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import matplotlib.pyplot as plt
        import psutil
    except ImportError as e:
        print("Missing required package. Please install:")
        print("pip install matplotlib psutil")
        sys.exit(1)
    
    benchmark = SortingBenchmark()
    benchmark.main()
