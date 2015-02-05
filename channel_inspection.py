#!/usr/bin/python2
#!/usr/bin/env python
#
# Copyright 2005,2007,2011 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

from gnuradio import gr, eng_notation
from gnuradio import blocks
from gnuradio import audio
from gnuradio import filter
from gnuradio import fft
from gnuradio import uhd
from gnuradio.eng_option import eng_option
from optparse import OptionParser
import sys
import math
import struct
import threading
from datetime import datetime
import matplotlib.pyplot as plt
import math
import csv
import time
from matplotlib.patches import Rectangle


sys.stderr.write("Warning: this may have issues on some machines+Python version combinations to seg fault due to the callback in bin_statitics.\n\n")

class ThreadClass(threading.Thread):
	def run(self):
		return

class tune(gr.feval_dd):
	"""
	This class allows C++ code to callback into python.
	"""
	def __init__(self, tb):
		gr.feval_dd.__init__(self)
		self.tb = tb

	def eval(self, ignore):
		"""
		This method is called from blocks.bin_statistics_f when it wants
		to change the center frequency.  This method tunes the front
		end to the new center frequency, and returns the new frequency
		as its result.
		"""

		try:
			# We use this try block so that if something goes wrong
			# from here down, at least we'll have a prayer of knowing
			# what went wrong.	Without this, you get a very
			# mysterious:
			#
			#	terminate called after throwing an instance of
			#	'Swig::DirectorMethodException' Aborted
			#
			# message on stderr.  Not exactly helpful ;)

			new_freq = self.tb.set_next_freq()
			
			# wait until msgq is empty before continuing
			while(self.tb.msgq.full_p()):
				#print "msgq full, holding.."
				time.sleep(0.1)
			
			return new_freq

		except Exception, e:
			print "tune: Exception: ", e


class parse_msg(object):
	def __init__(self, msg):
		self.center_freq = msg.arg1()
		self.vlen = int(msg.arg2())
		assert(msg.length() == self.vlen * gr.sizeof_float)

		# FIXME consider using NumPy array
		t = msg.to_string()
		self.raw_data = t
		self.data = struct.unpack('%df' % (self.vlen,), t)


class my_top_block(gr.top_block):

	def __init__(self):
		gr.top_block.__init__(self)

		usage = "usage: %prog [options] min_freq max_freq"
		parser = OptionParser(option_class=eng_option, usage=usage)
		parser.add_option("-a", "--args", type="string", default="",
						  help="UHD device device address args [default=%default]")
		parser.add_option("", "--spec", type="string", default=None,
					  help="Subdevice of UHD device where appropriate")
		parser.add_option("-A", "--antenna", type="string", default=None,
						  help="select Rx Antenna where appropriate")
		parser.add_option("-s", "--samp-rate", type="eng_float", default=1e6,
						  help="set sample rate [default=%default]")
		parser.add_option("-g", "--gain", type="eng_float", default=None,
						  help="set gain in dB (default is midpoint)")
		parser.add_option("", "--tune-delay", type="eng_float",
						  default=0.25, metavar="SECS",
						  help="time to delay (in seconds) after changing frequency [default=%default]")
		parser.add_option("", "--dwell-delay", type="eng_float",
						  default=0.25, metavar="SECS",
						  help="time to dwell (in seconds) at a given frequency [default=%default]")
		parser.add_option("-b", "--channel-bandwidth", type="eng_float",
						  default=6.25e3, metavar="Hz",
						  help="channel bandwidth of fft bins in Hz [default=%default]")
		parser.add_option("-l", "--lo-offset", type="eng_float",
						  default=0, metavar="Hz",
						  help="lo_offset in Hz [default=%default]")
		parser.add_option("-q", "--squelch-threshold", type="eng_float",
						  default=None, metavar="dB",
						  help="squelch threshold in dB [default=%default]")
		parser.add_option("-F", "--fft-size", type="int", default=None,
						  help="specify number of FFT bins [default=samp_rate/channel_bw]")
		parser.add_option("", "--real-time", action="store_true", default=False,
						  help="Attempt to enable real-time scheduling")
		parser.add_option("-v", "--vector_bw", type="eng_float", default=1e6, metavar="Hz",
						  help="Set number of channels for signal analysis output vector [default=%default]")
		parser.add_option("-d", "--display-graph", action="store_true", default=False,
						  help="Display fft data in a graph after each cycle")
		parser.add_option("-w", "--wb-threshold", type="eng_float",
						  default=None, metavar="dB",
						  help="wideband threshold in dB [default=%default]")


		(options, args) = parser.parse_args()
		if len(args) != 2:
			parser.print_help()
			sys.exit(1)

		self.channel_bandwidth = options.channel_bandwidth

		self.min_freq = eng_notation.str_to_num(args[0])
		self.max_freq = eng_notation.str_to_num(args[1])

		if self.min_freq > self.max_freq:
			# swap them
			self.min_freq, self.max_freq = self.max_freq, self.min_freq

		self.display_graph = options.display_graph

		if not options.real_time:
			realtime = False
		else:
			# Attempt to enable realtime scheduling
			r = gr.enable_realtime_scheduling()
			if r == gr.RT_OK:
				realtime = True
			else:
				realtime = False
				print "Note: failed to enable realtime scheduling"

		# build graph
		self.u = uhd.usrp_source(device_addr=options.args,
								 stream_args=uhd.stream_args('fc32'))

		# Set the subdevice spec
		if(options.spec):
			self.u.set_subdev_spec(options.spec, 0)

		# Set the antenna
		if(options.antenna):
			self.u.set_antenna(options.antenna, 0)
		
		self.u.set_samp_rate(options.samp_rate)
		self.usrp_rate = usrp_rate = self.u.get_samp_rate()
		
		self.lo_offset = options.lo_offset

		if options.fft_size is None:
			self.fft_size = int(self.usrp_rate/self.channel_bandwidth)
		else:
			self.fft_size = options.fft_size
		
		self.squelch_threshold = options.squelch_threshold
		self.wb_threshold = options.wb_threshold
	
		s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, self.fft_size)

		mywindow = filter.window.blackmanharris(self.fft_size)
		ffter = fft.fft_vcc(self.fft_size, True, mywindow, True)
		power = 0
		for tap in mywindow:
			power += tap*tap

		c2mag = blocks.complex_to_mag_squared(self.fft_size)

		# FIXME the log10 primitive is dog slow
		#log = blocks.nlog10_ff(10, self.fft_size,
		#						-20*math.log10(self.fft_size)-10*math.log10(power/self.fft_size))

		# Set the freq_step to 75% of the actual data throughput.
		# This allows us to discard the bins on both ends of the spectrum.

		self.freq_step = self.nearest_freq((0.75 * self.usrp_rate), self.channel_bandwidth)
		self.min_center_freq = self.min_freq + (self.freq_step/2) 
		nsteps = math.ceil((self.max_freq - self.min_freq) / self.freq_step)
		self.max_center_freq = self.min_center_freq + (nsteps * self.freq_step)

		self.next_freq = self.min_center_freq

		tune_delay	= max(0, int(round(options.tune_delay * usrp_rate / self.fft_size)))  # in fft_frames
		dwell_delay = max(1, int(round(options.dwell_delay * usrp_rate / self.fft_size))) # in fft_frames

		self.msgq = gr.msg_queue(1)
		self._tune_callback = tune(self)		# hang on to this to keep it from being GC'd
		stats = blocks.bin_statistics_f(self.fft_size, self.msgq,
										self._tune_callback, tune_delay,
										dwell_delay)

		# FIXME leave out the log10 until we speed it up
		#self.connect(self.u, s2v, ffter, c2mag, log, stats)
		self.connect(self.u, s2v, ffter, c2mag, stats)

		if options.gain is None:
			# if no gain was specified, use the mid-point in dB
			g = self.u.get_gain_range()
			options.gain = float(g.start()+g.stop())/2.0

		self.set_gain(options.gain)
		print "gain =", options.gain
	
		# initialize lists to hold data
		self.x_freq = []
		self.y_power = []
		self.sample_count = 0
		self.noise_floor_db = None

		# signal vector
		self.channel_width = options.vector_bw
		self.num_channels = int((self.max_freq - self.min_freq) / self.channel_width)

		self.signal_vector = [0] * self.num_channels


	def set_next_freq(self):
		target_freq = self.next_freq
		self.next_freq = self.next_freq + self.freq_step
		if self.next_freq >= self.max_center_freq:
			self.next_freq = self.min_center_freq
			
		
			self.y_power[:] = [x - self.noise_floor_db for x in self.y_power]


			combined_list = [self.x_freq,self.y_power]


			data_array = [list(m) for m in zip(*combined_list)]

			for item in data_array:	
				if (item[1] > self.squelch_threshold) and (item[0]*1e6 >= self.min_freq) and (item[0]*1e6 <= self.max_freq):
					#print datetime.now(), "center_freq", center_freq, "freq", freq, "power_db", power_db, "noise_floor_db", noise_floor_db
					channel_index = min(self.num_channels-1,max(0,int(math.floor((item[0]*1e6 - self.min_freq) / self.channel_width))))
					self.signal_vector[channel_index] = 1

			if self.wb_threshold is not None: 
				power_thres = self.wb_threshold
			else:
				power_thres = 35
			channel_bandwidth = 50e3
			signal_min_bandwidth = 1e6

			signal_id_array = self.signal_detection(data_array,power_thres,channel_bandwidth, signal_min_bandwidth)
			

			timestr = time.strftime("%Y%m%d-%H%M%S")
			filename_str = 'fft_samples/fft_' + timestr  + '.csv'

			with open(filename_str, 'wb') as f:
				writer = csv.writer(f)
				writer.writerows(combined_list)
	
			if self.display_graph:
				fig = plt.figure()
				plt.plot(self.x_freq, self.y_power, 'b-')
				
				for signal in signal_id_array:
					x1 = signal[0]/1e6
					width = (signal[1] - signal[0]) / 1e6
					y1 = 0
					height = 100
					currentAxis = plt.gca()
					currentAxis.add_patch(Rectangle((x1,y1), width, height, facecolor="green", alpha=0.5))


				#ax2 = fig.add_subplot(1,2,2)
				#ax2.bar(self.x_freq_bar, self.y_power_bar, int(self.channel_width))
				plt.show(block=False)
	


			else:
				# print "\rTo display the graph of the fft, use the -d option"
				pass
		
			# add wideband signals to the vector
			for signal in signal_id_array:
				start_channel = min(self.num_channels-1,max(0,int(math.floor((signal[0] - self.min_freq) / self.channel_width))))
				end_channel = min(self.num_channels-1,max(0,int(math.floor((signal[1] - self.min_freq) / self.channel_width))))

				for i in xrange(start_channel, end_channel + 1, 1):
					self.signal_vector[i] = 1		

 			# print out signal vector
			if self.squelch_threshold is not None:
				for i in xrange(0, self.num_channels):
					sys.stdout.write(str(self.signal_vector[i]))
				print " "
			else:
				print "\rTo print a signal vector, specify a squelch threshold using the -q option"


	
			#reset lists
			self.x_freq = []
			self.y_power = []
			self.sample_count = 0
			self.noise_floor_db = None

			self.signal_vector = [0] * self.num_channels
			self.y_power_bar = [0] * self.num_channels

		if not self.set_freq(target_freq):
			print "Failed to set frequency to", target_freq
			sys.exit(1)

		#sys.stdout.write("\rFrequency: %f MHz" % (target_freq / 1e6))
		#sys.stdout.flush()

		return target_freq


	def set_freq(self, target_freq):
		"""
		Set the center frequency we're interested in.

		Args:
			target_freq: frequency in Hz
		@rypte: bool
		"""
		
		r = self.u.set_center_freq(uhd.tune_request(target_freq, rf_freq=(target_freq + self.lo_offset),rf_freq_policy=uhd.tune_request.POLICY_MANUAL))
		if r:
			return True

		return False

	def set_gain(self, gain):
		self.u.set_gain(gain)
	
	def nearest_freq(self, freq, channel_bandwidth):
		freq = round(freq / channel_bandwidth, 0) * channel_bandwidth
		return freq

	def signal_detection(self, data_array,power_thres,channel_bandwidth, signal_min_bandwidth):

		spectrum_min = self.min_freq
		spectrum_max = self.max_freq
		freq_array = range(int(spectrum_min + channel_bandwidth*0.5), int(spectrum_max), int(channel_bandwidth))
		power_thres_array = [0] * len(freq_array)

		# print len(power_thres_array)
		
		#loop through frequency results and threshold into channel array

		try:
			for sample in data_array:
				#print sample
				if sample[1] > power_thres:
					index = min(len(freq_array)-1,max(0,int(math.floor((sample[0]*1e6 - spectrum_min)/channel_bandwidth))))
					#print index
					power_thres_array[index] = 1

		except Exception:
			print "Intial threshold index out of range"
			print "Index: %d, spectrum_min: %f, channel_bandwidth: %f" % (index,spectrum_min,channel_bandwidth)

		length = 0
		#print "Power thresholding done"
		length_thres = int(math.ceil(signal_min_bandwidth/channel_bandwidth))

		signal_id_array = [] #[signal start, signal end]
		
		try:

			for i in range(len(power_thres_array)):
				if power_thres_array[i] == 1:
					length += 1
				else:
					if length < length_thres:
						power_thres_array[i-length:i] = [0]*length
					else:
						signal_params = [freq_array[i-length],freq_array[i-1]]
						signal_id_array.append(signal_params)
					length = 0
		
		except Exception:
			print "Length thresholding index out of range"

		#for signal in signal_id_array:
		#	print "Signal start: %f, signal end: %f, signal width: %f" % (signal[0],signal[1],signal[1]-signal[0])
		
		return signal_id_array
		

def main_loop(tb):
	
	def bin_freq(i_bin, center_freq):
		#hz_per_bin = tb.usrp_rate / tb.fft_size
		freq = center_freq - (tb.usrp_rate / 2) + (tb.channel_bandwidth * i_bin)
		#print "freq original:",freq
		#freq = nearest_freq(freq, tb.channel_bandwidth)
		#print "freq rounded:",freq
		return freq
	
	bin_start = int(tb.fft_size * ((1 - 0.75) / 2))
	bin_stop = int(tb.fft_size - bin_start)

	while 1:

		# Get the next message sent from the C++ code (blocking call).
		# It contains the center frequency and the mag squared of the fft
		m = parse_msg(tb.msgq.delete_head())

		# m.center_freq is the center frequency at the time of capture
		# m.data are the mag_squared of the fft output
		# m.raw_data is a string that contains the binary floats.
		# You could write this as binary to a file.

		for i_bin in range(bin_start, bin_stop):

			center_freq = m.center_freq
			freq = bin_freq(i_bin, center_freq)
			#noise_floor_db = -174 + 10*math.log10(tb.channel_bandwidth)
			noise_floor_db = 10*math.log10(min(m.data)/tb.usrp_rate)
			power_db = 10*math.log10(m.data[i_bin]/tb.usrp_rate)# - noise_floor_db

			if tb.noise_floor_db is None or noise_floor_db < tb.noise_floor_db:
				tb.noise_floor_db = noise_floor_db

			tb.x_freq.append(freq/1e6)
			tb.y_power.append(power_db)

			#power_bar_index = int(math.floor((freq - tb.min_freq) / tb.channel_width))
			#tb.y_power_bar[power_bar_index] += power_db


if __name__ == '__main__':
	t = ThreadClass()
	t.start()

	tb = my_top_block()
	try:
		tb.start()
		main_loop(tb)

	except KeyboardInterrupt:
		pass
