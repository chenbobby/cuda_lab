package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"os"
	"os/exec"
	"time"

	_ "github.com/marcboeker/go-duckdb"
)

func main() {
	// Delete the target build directory.
	targetDir := "build"
	if err := os.RemoveAll(targetDir); err != nil {
		log.Fatal(err)
	}

	// Create the target build directory.
	if err := os.MkdirAll(targetDir, os.ModePerm); err != nil {
		log.Fatal(err)
	}

	// Profile the target program using nsys.
	// TODO: Add configuration for the nsys profile command.
	targetProgramFile := "../vec_add/build/vec_add_gpu"
	workloadName := "WORKLOAD"
	name := "nsys"
	args := []string{
		"profile",
		"--cuda-memory-usage=true",
		fmt.Sprintf("--output=%s/%s.nsys-rep", targetDir, workloadName),
		"--force-overwrite=true",
		targetProgramFile,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	if err := cmd.Run(); err != nil {
		log.Fatal(err)
	}

	// Export profiling data to an nsys SQLite file.
	// TODO: Add configuration for nsys stats command.
	name = "nsys"
	args = []string{
		"stats",
		"--report=cuda_gpu_mem_time_sum",
		"--report=cuda_gpu_mem_size_sum",
		"--force-export=true",
		fmt.Sprintf("%s/%s.nsys-rep", targetDir, workloadName),
	}
	ctx, cancel = context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cmd = exec.CommandContext(ctx, name, args...)
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	if err := cmd.Run(); err != nil {
		log.Fatal(err)
	}

	// Initialize the DuckDB in-memory database engine.
	db, err := sql.Open("duckdb", "")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// Initialize the DuckDB SQLite extension.
	_, err = db.Exec(`INSTALL sqlite; LOAD sqlite;`)
	if err != nil {
		log.Fatal(err)
	}

	// Initialize an SQLite database as a set of DuckDB tables.
	// TODO: Fix SQL injection vulnerability.
	_, err = db.Exec(fmt.Sprintf(`ATTACH '%s/%s.sqlite' (TYPE SQLITE); USE %s;`, targetDir, workloadName, workloadName))
	if err != nil {
		log.Fatal(err)
	}

	// Export all DuckDB tables to Parquet format.
	// TODO: Fix SQL injection vulnerability.
	_, err = db.Exec(fmt.Sprintf(`EXPORT DATABASE '%s' (FORMAT PARQUET);`, targetDir))
	if err != nil {
		log.Fatal(err)
	}

	log.Println("Exited successfully.")
}
