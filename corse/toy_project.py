import subprocess
import time

def greet_and_capture_output(cwd="/workspace/GPU_Profiling/kingcorsair"):
    result = subprocess.run(
            ["python", "/workspace/GPU_Profiling/kingcorsair/greet.py"],       
            cwd=cwd, 
            check=True,
            timeout=10,
            text=True,
            capture_output=True,
            )

    print(result.stdout)

def greet_and_write_to_file(cwd="/workspace/GPU_Profiling/kingcorsair"):
    start_within_function_call = time.perf_counter()
    with open("output.txt", "a") as output_file:
        for i in range(25):
            subprocess.run(
                ["python", "/workspace/GPU_Profiling/kingcorsair/greet.py"],       
                cwd=cwd, 
                timeout=10,
                text=True,
                stdout=output_file,
                stderr=output_file,
                )
    end_within_function_call = time.perf_counter()
    execution_time_within_function_call = end_within_function_call - start_within_function_call
    print(f"Execution time within function call: {execution_time_within_function_call:.6f} seconds")

#greet_and_capture_output()
start_before_function_call = time.perf_counter()
greet_and_write_to_file()
end_after_function_call = time.perf_counter()
execution_time = end_after_function_call - start_before_function_call
print(f"Execution time: {execution_time:.6f} seconds")