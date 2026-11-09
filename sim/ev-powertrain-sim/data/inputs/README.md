# data/inputs/: real inputs to the drive-cycle runs
* `udds.txt`: EPA Urban Dynamometer Driving Schedule (time s, target speed mph), 1,372 lines.
* `ds_tum/track_{0,1,10,100,103}.csv`: TUM Tesla Model 3 CAN traces, CC BY 4.0, Rosenberger et al., doi:10.14459/2024mp1735471 (10 Hz). `Checksums.sha512` is the publisher's file. SHA-256 of `track_0.csv` equals the value recorded in `../tum_subset/SOURCE.json`.
* Used by `run_cycle_matrix.py` (UDDS 120 s, Tesla track 0 for 150 s) and `tools/storage_bench.py`. Column meaning, schema differences between tracks and which columns are trusted: `docs/DATASET_INVENTORY.md`.
* Tracks 0 and 1 have an untrustworthy `rear_motor_torque`; the runs use speed, not that torque.
