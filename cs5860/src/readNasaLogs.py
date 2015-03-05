import pandas as pd

f0 = "../data/NASA_test"
f1 = "../data/access_log_Aug95"
f2 = "../data/access_log_Jul95"
fp = "../data/nasa_access_log.pkl"

# src | date | time | tz | get/post | page | protocol | svr resp | size
colheaders = ["src","date","time","tz","getpost","page","protocol","svrresp","size"]
logentry=[]
lerr=[]
for fn in [f1, f2]:
	for line in open(fn,'r'):
		# ensure only two quote marks in entry
		if len(line.split('"')) <> 3:
			lerr.append(line)
			continue
		try:
			# get/post first entry in quoted string
			getpost = line.split('"')[1].split(' ')[0]
			try:
				# readable URL in second term
				url = line.split('"')[1].split(' ')[1]
				try:
					# readable protocol in second term
					prtcl = line.split('"')[1].split(' ')[2]
				except:
					prtcl = ""
			except:
				lerr.append(line)
				continue
		except:
			lerr=lerr.append(line)
			continuel
		logentry.append([
			line.split(' ')[0],
			line.split(' ')[3].replace('[','').split(':')[0],
			':'.join(line.split(' ')[3].replace('[','').split(':')[1:]),
			line.split(' ')[4].replace(']',''),
			getpost,
			url,
			prtcl,
			line.split('"')[2].split(' ')[1],
			line.split('"')[2].split(' ')[2].replace('\n',''),
			])

len(logentry) # 3461567 both | 1891695 in Jul | 1569872 in Aug
len(lerr) # 46 both | 20 in Jul | 26 in Aug
for l in lerr:
	print l

# Data Frame
dflog = pd.DataFrame(logentry, columns=colheaders)
dflog.to_pickle(fp)
