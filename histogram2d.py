from math import *
from random import *

# All the histograms use logarithmic binning. Below, the function histogram2d_cond produces
# conditional distributions in addition to raw distributions.

def histogram2d(mux,muy,infile,outfile):
	h={}
	n=0.
	x_l=[]
	y_l=[]
	xy_l=[]
	input=open(infile,'r')
	for line in input:
		sl=line.split()
		x_l.append(float(sl[0]))
		y_l.append(float(sl[1].strip()))
		xy_l.append((float(sl[0]),float(sl[1].strip())))
	x_l.sort()
	y_l.sort()
	input.close()
	xo=x_l[0]
	xf=x_l[-1]
	yo=y_l[0]
	yf=y_l[-1]
	ixmax=int(floor(log(xf/xo)/log(mux)))
	iymax=int(floor(log(yf/yo)/log(muy)))
	print "MINS AND MAXS:",xo,xf,yo,yf,ixmax,iymax

	for n in range(len(xy_l)):
		ix=floor(log(xy_l[n][0]/xo)/log(mux))
		iy=floor(log(xy_l[n][1]/yo)/log(muy))
		h[(ix,iy)]=h.get((ix,iy),0)+1
	
	sum=0.
	for i in range(ixmax+1):
		for j in range(iymax+1):
			if h.has_key((i,j)):
				h[(i,j)]=h[(i,j)]/(((mux-1.)*mux**i)*((muy-1.)*muy**j))
				sum=sum+h[(i,j)]
	for i in range(ixmax+1):
		for j in range(iymax+1):
			if h.has_key((i,j)):
				h[(i,j)]=h[(i,j)]/sum
	check_sum=0.
	output=open(outfile,'w')
	for i in range(ixmax+1):
		for j in range(iymax+1):
			if h.has_key((i,j)):
				output.write('%s %s %s\n'%(xo*mux**i,yo*muy**j,h[(i,j)]))
				check_sum=check_sum+h[(i,j)]
	output.close()
	print "NORMALIZATION",check_sum
	return(h)

def histogram1d(mux,infile,pdf_outfile):
	h={}
	n=0.
	x_l=[]
	input=open(infile,'r')
	for line in input:
		sl=line.split()
		x_l.append(float(sl[0].strip()))
	input.close()
	x_l.sort()
	xo=x_l[0]
	xf=x_l[-1]
	ixmax=int(floor(log(xf/xo)/log(mux)))
	print "MINS AND MAXS:",xo,xf,ixmax

	for n in range(len(x_l)):
		ix=floor(log(x_l[n]/xo)/log(mux))
		h[ix]=h.get(ix,0)+1
	
	sum=0.
	for i in range(ixmax+1):
		if h.has_key(i):
			h[i]=h[i]/((mux-1.)*mux**i)
			sum=sum+h[i]
	for i in range(ixmax+1):
		if h.has_key(i):
			h[i]=h[i]/sum
	check_sum=0.
	output=open(pdf_outfile,'w')
	for i in range(ixmax+1):
		if h.has_key(i):
			output.write('%s %s\n'%(xo*mux**i,h[i]))
			check_sum=check_sum+h[i]
	output.close()
#	ch={}
#	for i in range(ixmax+1):
#		if h.has_key(i):
#			ch[i]=ch.get(i-1,0)+h[i]
#	output=open(ccdf_outfile,'w')
#	for i in range(ixmax+1):
#		if ch.has_key(i):
#			output.write('%s %s\n'%(xo*mux**i,1.-ch[i]))
#	output.close()
	print "NORMALIZATION",check_sum
	return(h)

def histogram1d_v2(mux,infile,pdf_outfile):
	h={}
	n=0.
	x_l=[]
	input=open(infile,'r')
	for line in input:
		sl=line.split()
		x_l.append(float(sl[0].strip()))
	input.close()
	x_l.sort()
	xo=x_l[0]
	xf=x_l[-1]
	ixmax=int(floor(log(xf/xo)/log(mux)))
	print "MINS AND MAXS:",xo,xf,ixmax

	for n in range(len(x_l)):
		ix=floor(log(x_l[n]/xo)/log(mux))
		h[ix]=h.get(ix,0)+1
	
	sum=0.
	for i in range(ixmax+1):
		if h.has_key(i):
			h[i]=h[i]/(xo*(mux-1.)*mux**i)
			sum=sum+h[i]
	for i in range(ixmax+1):
		if h.has_key(i):
			h[i]=h[i]/sum
	check_sum=0.
	output=open(pdf_outfile,'w')
	for i in range(ixmax+1):
		if h.has_key(i):
			output.write('%s %s\n'%(xo*mux**i,h[i]))
			check_sum=check_sum+h[i]
	output.close()
#	ch={}
#	for i in range(ixmax+1):
#		if h.has_key(i):
#			ch[i]=ch.get(i-1,0)+h[i]
#	output=open(ccdf_outfile,'w')
#	for i in range(ixmax+1):
#		if ch.has_key(i):
#			output.write('%s %s\n'%(xo*mux**i,1.-ch[i]))
#	output.close()
	print "NORMALIZATION",check_sum
	return(h)

# THIS IS ANOTHER LOG BIN HISTOGRAM, FOCUSED ON n THE NUMBER OF BINS
# infile (1D): LIST OF VALUES TO BE BINNED
def histogram1d_nbin(n,infile,outfile):
	input=open(infile,'r')
	xl=[]
	for line in input:
		sl=float(line.strip())
		xl.append(sl)
	input.close()
	xl.sort()
	xo=xl[0]
	xf=xl[-1]
	lmu=log10(xf/xo)/n
	mu=10**lmu
	h={}
	for i in range(n):
		h[i]=0.
	for x in xl:
		if x==xf:
			h[n-1]=h[n-1]+1
		else:
			i=floor(log10(x/xo)/lmu)
			h[i]=h[i]+1
	sum=0.
	for i in range(n):
		h[i]=h[i]/mu**i
		sum=sum+h[i]
	for i in range(n):
		h[i]=h[i]/sum
	output=open(outfile,'w')
	for i in range(n):
		if h[i]>0:
			output.write('%s %s\n'%(xo*mu**i,h[i]))
	output.close()
	return(h)

# NEXT ROUTINE PRODUCES ALL NECESSARY FILES FOR 3D HISTOGRAMS. IN DETAIL:
# outfile(3D) IS FILE FOR THE SIMPLE CONDITIONAL PROBABILITY
# outfile_normalized(3D) IS FILE FOR CONDITIONAL PROBABILITY DIVIDED BY PROBABILITY OF MODE
# outfile_modes(2D) IS FILE OF LOCATION X AND Y OF MODE
# outfile_modes_probs(3D) IS FILE OF LOCATION XY OF MODE V ITS CONDITIONAL PROBABILITY
# outfile_modes_joint_probs(3D) IS FILE OF LOCATION XY OF MODE V ITS RAW (JOINT) PROBABILITY
# infile IS A TWO-COLUMN FILE
def histogram2d_cond(mux,muy,infile,outfile,outfile_normalized,outfile_modes,outfile_modes_probs,outfile_modes_joint_probs):
	h={}
	h_cond={}
	n=0.
	x_l=[]
	y_l=[]
	xy_l=[]
	input=open(infile,'r')
	for line in input:
		sl=line.split()
		x_l.append(float(sl[0]))
		y_l.append(float(sl[1].strip()))
		xy_l.append((float(sl[0]),float(sl[1].strip())))
	x_l.sort()
	y_l.sort()
	input.close()
	xo=x_l[0]
	xf=x_l[-1]
	yo=y_l[0]
	yf=y_l[-1]
	ixmax=int(floor(log(xf/xo)/log(mux)))
	iymax=int(floor(log(yf/yo)/log(muy)))
	print "MINS AND MAXS:",xo,xf,yo,yf,ixmax,iymax

	for n in range(len(xy_l)):
		ix=floor(log(xy_l[n][0]/xo)/log(mux))
		iy=floor(log(xy_l[n][1]/yo)/log(muy))
		h[(ix,iy)]=h.get((ix,iy),0)+1
	
	sum=0.
	sum_marg={}
	for i in range(ixmax+1):
		for j in range(iymax+1):
			if h.has_key((i,j)):
				h[(i,j)]=h[(i,j)]/(((mux-1.)*mux**i)*((muy-1.)*muy**j))
				sum=sum+h[(i,j)]
				sum_marg[i]=sum_marg.get(i,0.)+h[(i,j)]
	h_cond_star={}
	jstar={}
	for i in range(ixmax+1):
		for j in range(iymax+1):
			if h.has_key((i,j)):
				h_cond[(i,j)]=h[(i,j)]/sum_marg[i]
				h[(i,j)]=h[(i,j)]/sum
				h_cond_star[i]=h_cond_star.get(i,0.)
				curr_max=h_cond_star[i]
				h_cond_star[i]=max(h_cond_star[i],h_cond[(i,j)])
				if h_cond_star[i]!=curr_max:
					curr_max=h_cond_star[i]
					jstar[i]=j
	check_sum=0.
	output=open(outfile,'w')
	output_n=open(outfile_normalized,'w')
	output_modes=open(outfile_modes,'w')
	output_modes_and_probs=open(outfile_modes_probs,'w')
	output_modes_and_joint_probs=open(outfile_modes_joint_probs,'w')
	for i in range(ixmax+1):
		if jstar.has_key(i):
			output_modes.write('%s %s\n'%(xo*mux**i,yo*muy**jstar[i]))
			output_modes_and_probs.write('%s %s %s\n'%(xo*mux**i,yo*muy**jstar[i],h_cond_star[i])) 
			output_modes_and_joint_probs.write('%s %s %s\n'%(xo*mux**i,yo*muy**jstar[i],h[(i,jstar[i])])) 
		for j in range(iymax+1):
			if h.has_key((i,j)):
				output.write('%s %s %s\n'%(xo*mux**i,yo*muy**j,h_cond[(i,j)]))
				output_n.write('%s %s %s\n'%(xo*mux**i,yo*muy**j,h_cond[(i,j)]/h_cond_star[i]))
				check_sum=check_sum+h[(i,j)]
	output.close()
	output_n.close()
	output_modes.close()
	output_modes_and_probs.close()
	output_modes_and_joint_probs.close()
	print "NORMALIZATION",check_sum
	return(h_cond)

def max(x,y):
	if x>y:
		return(x)
	else:
		return(y)

def histogram1d_linear(nb,infile,pdf_outfile):
	h={}
	n=0.
	x_l=[]
	input=open(infile,'r')
	for line in input:
		sl=line.split()
		x_l.append(float(sl[0].strip()))
	input.close()
	x_l.sort()
	xo=x_l[0]
	xf=x_l[-1]
	dx=(xf-xo)/nb
	print "MINS AND MAXS:",xo,xf,dx

	for n in range(len(x_l)):
		ix=floor(x_l[n]-xo)
		if ix<nb-1:
			h[ix]=h.get(ix,0)+1
		else:
			h[ix-1]=h.get(ix-1,0)+1
	
	sum=0.
	for i in range(nb):
		if h.has_key(i):
			sum=sum+h[i]
	for i in range(nb):
		if h.has_key(i):
			h[i]=h[i]/sum
	check_sum=0.
	output=open(pdf_outfile,'w')
	for i in range(nb):
		if h.has_key(i):
			output.write('%s %s\n'%(xo+(i+0.5)*dx,h[i]))
			check_sum=check_sum+h[i]
	output.close()
	print "NORMALIZATION",check_sum
	return(h)

