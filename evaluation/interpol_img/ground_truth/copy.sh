dest_pre='./test/pose/248/'
while read file; do
    echo 'Copying:' $file
    dest=`echo $file | rev | cut -d_ -f 1 | rev`
    # echo $dest
    `cp $file $dest_pre$dest`
done<files
