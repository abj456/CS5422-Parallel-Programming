# #! /bin/bash
# module load mpi nsys
# N=4
# R=3
# nsys stats \
#     -r mpi_event_trace,nvtx_pushpop_trace \
#     --timeunit sec \
#     --format csv \
#     --force-export=true \
#     --output . \
#     "./nsys_reports_n$N/rank_$R.nsys-rep"
#     $@

#! /bin/bash
module load mpi nsys

for report_dir in ./nsys_reports_n*; do
    for report_file in "$report_dir"/*.nsys-rep; do
        nsys stats \
            -r mpi_event_trace,nvtx_pushpop_trace \
            --timeunit sec \
            --format csv \
            --force-export=true \
            --output "$report_file.stats" \
            "$report_file" \
            "$@"
    done
done
