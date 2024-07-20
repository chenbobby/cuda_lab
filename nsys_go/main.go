package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path"
	"time"

	"github.com/chenbobby/cuda_lab/nsys_report_sqlite/events"
	_ "github.com/marcboeker/go-duckdb"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
	log.Println("Exited successfully.")
}

func run() error {
	// Configuration
	targetDirPath := "build"
	workloadCommand := "../vec_add/build/vec_add_gpu"
	nsysRepReportOutputPath := "build/profile/report.nsys-rep"
	sqliteReportOutputPath := "build/profile/report.sqlite"
	duckdbReportOutputPath := "build/profile/out.duckdb"

	// Delete the target build directory.
	if err := os.RemoveAll(targetDirPath); err != nil {
		return err
	}

	// Create the target build directory.
	if err := os.MkdirAll(targetDirPath, os.ModePerm); err != nil {
		return err
	}

	// Delete the target report build directory.
	targetReportDirPath := path.Join(targetDirPath, "profile")
	if err := os.RemoveAll(targetReportDirPath); err != nil {
		return err
	}

	// Create the target report build directory.
	if err := os.MkdirAll(targetReportDirPath, os.ModePerm); err != nil {
		return err
	}

	// Profile the workload using nsys.
	// TODO: Add configuration for the nsys profile command.
	name := "nsys"
	args := []string{
		"profile",
		"--cuda-memory-usage=true",
		fmt.Sprintf("--output=%s", nsysRepReportOutputPath),
		"--force-overwrite=true",
		workloadCommand,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	if err := cmd.Run(); err != nil {
		return err
	}

	// Export profiling data to an nsys SQLite file.
	// TODO: Add configuration for nsys export command.
	name = "nsys"
	args = []string{
		"export",
		"--type=sqlite",
		"--lazy=false",
		fmt.Sprintf("--output=%s", sqliteReportOutputPath),
		"--force-overwrite=true",
		nsysRepReportOutputPath,
	}
	ctx, cancel = context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cmd = exec.CommandContext(ctx, name, args...)
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	if err := cmd.Run(); err != nil {
		return err
	}

	// Initialize the DuckDB database engine.
	db, err := sql.Open("duckdb", duckdbReportOutputPath)
	if err != nil {
		return err
	}
	defer db.Close()

	// Initialize the DuckDB SQLite extension.
	_, err = db.Exec(`INSTALL sqlite; LOAD sqlite;`)
	if err != nil {
		return err
	}

	// Initialize an SQLite database as a set of DuckDB tables.
	// TODO: Fix SQL injection vulnerability.
	_, err = db.Exec(fmt.Sprintf(`ATTACH '%s' (TYPE SQLITE); USE report;`, sqliteReportOutputPath))
	if err != nil {
		return err
	}

	// Read all table names from workload profile SQLite file.
	rows, err := db.Query(`SHOW TABLES;`)
	if err != nil {
		return err
	}

	tableNames := make([]string, 0)
	for rows.Next() {
		var tableName string
		if err := rows.Scan(&tableName); err != nil {
			return err
		}
		tableNames = append(tableNames, tableName)
	}

	// Switch to the DuckDB database.
	// TODO: Use randomized tmp database names.
	_, err = db.Exec("USE out;")
	if err != nil {
		return err
	}

	// Create DuckDB tables from SQLite tables.
	for _, tableName := range tableNames {
		// Create the DuckDB table from the SQLite table.
		_, err = db.Exec(fmt.Sprintf(`CREATE TABLE out.%s AS SELECT * FROM report.%s;`, tableName, tableName))
		if err != nil {
			return err
		}
	}

	if err := db.Close(); err != nil {
		return err
	}

	// Open new read-only connection
	db, err = sql.Open("duckdb", duckdbReportOutputPath)
	if err != nil {
		return err
	}
	defer db.Close()

	return query(db)
}

func query(db *sql.DB) error {
	// List all tables in the DuckDB database.
	rows, err := db.Query(`SHOW TABLES;`)
	if err != nil {
		return err
	}

	for rows.Next() {
		var tableName string
		if err := rows.Scan(&tableName); err != nil {
			return err
		}
		log.Println(tableName)
	}

	queries := events.New(db)

	events, err := queries.GetCUDAMemoryUsageEvents(context.Background())
	if err != nil {
		return err
	}

	for _, event := range events {
		log.Println(event)
	}
	return nil
}
