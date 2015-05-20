cat NASA_access_log_Jul95 NASA_access_log_Aug95 | cut -d '"' -f2 | cut -d ' ' -f2 > x
sort x > y; rm x  
uniq -c y > z; rm y  
sort -bnr z > freq.count; rm z  

31059   62118 1323140 freq.count

# create directory tree: ROOT
cat NASA_access_log_Jul95 NASA_access_log_Aug95 | cut -d '"' -f2 | cut -d ' ' -f2 | cut -d '/' -f1-2 > x
sort x > y; rm x  
uniq -c y > z; rm y  
sort -bnr z > level1.count; rm z  

cat NASA_access_log_Jul95 NASA_access_log_Aug95 | cut -d '"' -f2 | cut -d ' ' -f2 | cut -d '/' -f1-3 > x
sort x > y; rm x  
uniq -c y > z; rm y  
sort -bnr z > level2.count; rm z  

cat NASA_access_log_Jul95 NASA_access_log_Aug95 | cut -d '"' -f2 | cut -d ' ' -f2 | cut -d '/' -f1-4 > x
sort x > y; rm x  
uniq -c y > z; rm y  
sort -bnr z > level3.count; rm z  

grep -v "\.\|\?" freq.count.trunc | awk '{split($0,a," "); print a[2],a[1]}' | sort > x

# get sources
cat NASA_access_log_Jul95 NASA_access_log_Aug95 | cut -d ' ' -f1  | sort -u > src.ip

rm src_tgt_tagged
while read i; do
	src=`echo $i | cut -d ' ' -f1`
	page=`echo $i | cut -d '"' -f2 | cut -d ' ' -f2 `
	cnt=`grep " ${page}$" freq.count | cut -b1-8`
	if [ ${cnt} -gt 20 ]; then
		echo ${src} ${page} "pass" >> src_tgt_tagged
	else
		echo ${src} ${page} "fail" >>  src_tgt_tagged
	fi
done < freq.count



