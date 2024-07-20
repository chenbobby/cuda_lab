package main

import (
	"database/sql"
	"fmt"
	"log"
	"os"

	_ "github.com/marcboeker/go-duckdb"
)

func main() {
	targetDir := "build"

	// Delete the target build directory.
	if err := os.RemoveAll(targetDir); err != nil {
		log.Fatal(err)
	}

	// Create the target build directory.
	if err := os.MkdirAll(targetDir, os.ModePerm); err != nil {
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
	inputNsysSqliteFile := "../vec_add/report/vec_add_gpu.sqlite"
	namespace := "vec_add_gpu"
	// TODO: Fix SQL injection vulnerability.
	_, err = db.Exec(fmt.Sprintf(`ATTACH '%s' (TYPE SQLITE); USE %s;`, inputNsysSqliteFile, namespace))
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
