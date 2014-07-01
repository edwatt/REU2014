#!/usr/bin/env python

"""
Constant random frequency hop - Detects and avoids interference by changing to a different channel subset
"""
from gnuradio import gr
from gnuradio import analog
from gnuradio import uhd
from time import sleep
from gnuradio import filter
from gnuradio import fft
from gnuradio import blocks
from gnuradio import audio
from gnuradio import eng_notation
from optparse import OptionParser
import sys
import math
import struct
from gnuradio.eng_option import eng_option
import threading
import random
from datetime import datetime

MAX_RATE = 1000e6


class ThreadClass(threading.Thread):
	def run(self):
		return

class tune(gr.feval_dd):
	"""
	This class allows C++ code to callback into python
	"""

	def __init__(self, tb):
		gr.feval_dd.__init__(self)
		self.tb = tb

	def eval(self, ignore):
		"""
		This method is called from blocks.bin_statistics_f when it wants
		to change the center frequency. This method tunes the fron
		end to the new center frequency, and returns the new frequency
		as its result.
		"""

		try:
			# We use this try block so that if something goes wrong
			# from here on down, at least we'll have a prayer of knowing
			# what went wrong. Without this, you get a very
			# mysterious:
			#
			#   terminate called after throwing an instance of
			#	'Swig::DirectorMethodException' Aborted
			#
			# message on stderr. Not exactly helpful ;)

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

		t = msg.to_string()
		self.raw_data = t
		self.data = struct.unpack('%df' % (self.vlen,), t)


class build_block(gr.top_block):
	def __init__(self):
		gr.top_block.__init__(self)

		usage = "usage: %prog [options] min_freq max_freq"
		parser = OptionParser(option_class=eng_option, usage=usage)
		parser.add_option("-a", "--args", type="string", default="",
						  help="UHD device address args [default=%default]")
		parser.add_option("", "--spec", type="string", default=None,
						  help="Subdevice of UHD device where appropriate")
		parser.add_option("-R", "--rx-antenna", type="string", default="RX2",
						  help="select RX antenna where appropriate")
		parser.add_option("-T", "--tx-antenna", type="string", default="TX/RX",
						  help="select TX antenna where appropriate")
		parser.add_option("-s", "--samp-rate", type="eng_float", default=1e6,
						  help="set sample rate [default=%default]")
		parser.add_option("-g","--gain", type="eng_float", default=None,
						  help="set gain in dB (default is midpoint)")
		parser.add_option("","--tune-delay", type="eng_float", default=0.25,
						  metavar="SECS", help="time to delay (in seconds) after changing frequency [default=%default]")
		parser.add_option("","--dwell-delay", type="eng_float", default=0.25,
						  metavar="SECS", help="time to delay (in seconds) at a given frequency [default=%default]")
		parser.add_option("-b","--channel-bandwidth", type="eng_float", default=6.25e3,
						  metavar="Hz", help="channel bandwidth of fft bins in Hz [default=%default]")
		parser.add_option("-l", "--lo-offset", type="eng_float", default=0,
						  metavar="Hz", help="lo_offset in Hz [default=%default]")
		parser.add_option("-q", "--squelch-threshold", type="eng_float", default=None,
						  metavar="dB", help="squelch threshold in dB [default=%default]")
		parser.add_option("-F", "--fft-size", type="int", default=None,
						  help="specify the number of FFT bins [default=samp_rate/channel_bw]")
		parser.add_option("", "--real-time", action="store_true", default=False,
						  help="Attempt to enable real-time scheduling")
		parser.add_option("-w","--tx-bandwidth", type="eng_float", default=6e6,
						  metavar="Hz", help="transmit frequency bandwidth [default=%default]")

		(options, args) = parser.parse_args()
		if len(args) != 2:
			parser.print_help()
			sys.exit(1)

		self.channel_bandwidth = options.channel_bandwidth #fft channel bandwidth
		self.tx_bandwidth = options.tx_bandwidth

		self.min_freq = eng_notation.str_to_num(args[0])
		self.max_freq = eng_notation.str_to_num(args[1])

		if self.min_freq < 1e6: self.min_freq *= 1e6
		if self.max_freq < 1e6: self.max_freq *= 1e6

		self.n_channel_grps = 5 #number of channel groups
		self.curr_channel_grp = 0
		self.n_channels = 5 #number of channels / channel group		
		self.curr_channel = 0

		self.tx_guard_band = 4e6


		if self.min_freq > self.max_freq:
			#swap them
			self.min_freq, self.max_freq = self.max_freq, self.min_freq

		self.channel_grp_bw = (self.max_freq-self.min_freq) / self.n_channel_grps
		self.tx_channel_bw = self.channel_grp_bw / self.n_channels # tx channel bw

		if not options.real_time:
			real_time = False
		else:
			#Attempt to enable realtime scheduling
			r = gr.enable_realtime_scheduling()
			if r == gr.RT_OK:
				realtime=True
			else:
				realtime=False
				print "Note: failed to enable realtime scheduling"

		#build graph
		self.u_rx = uhd.usrp_source(device_addr=options.args, stream_args=uhd.stream_args('fc32'))

		#Set the subdevice spec
		if options.spec:
			self.u_rx.set_subdev_spec(options.spec, 0)

		#Set the antenna
		if options.rx_antenna:
			self.u_rx.set_antenna(options.rx_antenna, 0)

		self.u_rx.set_samp_rate(options.samp_rate)
		self.usrp_rate = usrp_rate = self.u_rx.get_samp_rate()

		self.lo_offset = options.lo_offset

		if options.fft_size is None:
			self.fft_size = int(self.usrp_rate/self.channel_bandwidth) 
		else:
			self.fft_size = options.fft_size

		print "FFT Size: %d, USRP Samp Rate: %d, Channel Bandwidth: %d" % (self.fft_size, self.usrp_rate, self.channel_bandwidth)	

		self.squelch_threshold = options.squelch_threshold

		s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, self.fft_size)

		mywindow=filter.window.blackmanharris(self.fft_size)
		ffter = fft.fft_vcc(self.fft_size, True, mywindow, True)

		c2mag = blocks.complex_to_mag_squared(self.fft_size)

		tune_delay = max(0, int(round(options.tune_delay * usrp_rate / self.fft_size))) # in fft_frames
		dwell_delay = max(1, int(round(options.dwell_delay * usrp_rate / self.fft_size))) # in fft frames

		self.msgq = gr.msg_queue(1)
		self._tune_callback = tune(self)   #hang on to this to keep it from being GC'd
	
		stats = blocks.bin_statistics_f(self.fft_size, self.msgq, self._tune_callback, tune_delay, dwell_delay)

		self.connect(self.u_rx, s2v, ffter, c2mag, stats)

		#transmit chain


		d = uhd.find_devices(uhd.device_addr(options.args))
		if d:

			uhd_type = d[0].get('type')
			print "\nFound '%s'" % uhd_type
		else:
			print "\nNo device found"
			self.u_tx = None
			return

		#check version of USRP and set num_channels

		if uhd_type == "usrp":
			tx_nchan = 2
			rx_nchan = 2
		else:
			tx_nchan = 1
			rx_nchan = 1

		#setup transmit chain (usrp sink, signal source)

		#usrp sink
		stream_args = uhd.stream_args('fc32', channels = range(tx_nchan))
		self.u_tx = uhd.usrp_sink(device_addr=options.args, stream_args=stream_args)
		self.u_tx.set_samp_rate(self.usrp_rate)

		if options.tx_antenna:
			self.u_tx.set_antenna(options.tx_antenna,0)

        #analog signal source - sig_source_c(sampling_freq,waveform, wave_freq, ampl, offset=0)
		self.tx_src0 = analog.sig_source_c(self.u_tx.get_samp_rate(), analog.GR_CONST_WAVE, 0, 1.0, 0)
	
		#connect blocks

		self.connect(self.tx_src0, self.u_tx)

		if options.gain is None:
			# if no gain was specified use the midpoint in dB
			g = self.u_rx.get_gain_range()
			options.gain = float(g.start()+g.stop())/2.0

		self.set_gain(options.gain)
		print "gain =", options.gain

		#initialize transmission parameters
		self.set_channel_group(3)
		self.step_count = 0

	def set_next_freq(self):
		target_freq = self.next_freq
		self.next_freq = self.next_freq + self.freq_step
		if self.next_freq >= self.max_center_freq:
			self.next_freq = self.min_center_freq
			# print "Step Count: %d" % self.step_count
			# self.step_count = 0i
			print "Channel Group scan complete. Starting scan again."
			sys.stdout.flush()

		freq_diff = abs(target_freq - self.tx_freq)
		print "Target Freq: %d MHz, Diff: %d MHz, TX BW: %f MHz, Freq Step: %f MHz, Min dist: %f MHz" % ((target_freq / 1e6), (freq_diff / 1e6),(self.tx_bandwidth / 1e6 ),(self.freq_step / 1e6), ( (self.tx_bandwidth / 2 + self.freq_step / 2)/1e6 ))

		if freq_diff < ( self.tx_bandwidth / 2 + self.freq_step / 2 ):
			self.tx_off()
		else:
			self.tx_on()

		if not self.set_rx_freq(target_freq):
			print "Failed to set frequency to", target_freq
			sys.exit(1)

		return target_freq


	def set_channel_group(self,channel_grp):
		self.curr_channel_grp = channel_grp
		#set new min and max freq for spectrum sense
		self.spec_min_freq = self.min_freq + self.channel_grp_bw * self.curr_channel_grp
		self.spec_max_freq = self.spec_min_freq + self.channel_grp_bw
		#set initial tx freq
		self.set_tx_channel(random.randint(0,(self.n_channels - 1)))
		
		self.update_spec_sense_parameters()

	def update_spec_sense_parameters(self):
		self.freq_step = self.nearest_freq((0.75 * self.usrp_rate), self.channel_bandwidth)
		self.min_center_freq = self.spec_min_freq + (self.freq_step / 2)
		print "Min center freq: %f" % self.min_center_freq
		nsteps = math.ceil((self.spec_max_freq - self.spec_min_freq) / self.freq_step)
		self.max_center_freq = self.min_center_freq + (nsteps * self.freq_step)
		print "Max center freq: %f" % self.max_center_freq
		self.next_freq = self.min_center_freq

	def set_tx_channel(self, channel):
		self.curr_channel = channel
		target_freq = self.spec_min_freq + (self.curr_channel + 0.5) * self.tx_channel_bw
		self.tx_freq = target_freq 
		self.u_tx.set_center_freq(target_freq)	
		self.tx_freq_min = target_freq - self.tx_guard_band
		self.tx_freq_max = target_freq + self.tx_guard_band
		print "\n TX Freq: %f" % target_freq


	def set_rx_freq(self, target_freq):
		"""
		Set the cetner frequency we are interested in.

		Args:
			target frequency in Hz
		@rypte: bool
		"""

		r = self.u_rx.set_center_freq(uhd.tune_request(target_freq, rf_freq=(target_freq + self.lo_offset),rf_freq_policy=uhd.tune_request.POLICY_MANUAL))
		if r:
			return True
		
		return False

	def set_gain(self, gain):
		self.u_rx.set_gain(gain)
		self.u_tx.set_gain(gain * 1.75)

	def nearest_freq(self, freq, channel_bandwidth):
		freq = round(freq / channel_bandwidth, 0) * channel_bandwidth
		return freq	
	def tx_on(self):
		if self.tx_src0.amplitude() < 1.0:
			self.tx_src0.set_amplitude(1.0)
			print "TX ON"
	def tx_off(self):
		if self.tx_src0.amplitude() > 0.0:
			self.tx_src0.set_amplitude(0.0)
			print "TX OFF"

def main_loop(tb):

	def bin_freq(i_bin, center_freq):
		freq = center_freq - (tb.usrp_rate / 2) + (tb.channel_bandwidth * i_bin)
		return freq

	bin_start = int(tb.fft_size * ((1 - 0.75) / 2))
	bin_stop = int(tb.fft_size - bin_start)

	#create transmission hopping thread
	#start thread

	while 1:

		# Get the next message sent from the C++ code (blocking call)
		# It contains the center frequency and the mag squared of the fft
		m = parse_msg(tb.msgq.delete_head())

		# m.center_freq is the center frequency at the time of capture
		# m.data are the mag_squared of the fft output
		# m.raw_data is a string that contains the binary floats
		# You could write this as a binary to a file

		for i_bin in range(bin_start, bin_stop):
			center_freq = m.center_freq
			freq = bin_freq(i_bin, center_freq)
			noise_floor_db = 10*math.log10(min(m.data)/tb.usrp_rate)
			power_db = 10*math.log10(m.data[i_bin]/tb.usrp_rate) - noise_floor_db
			tb.step_count += 1
			# print datetime.now(), "center_freq", center_freq, "freq", freq, "power_db", power_db, "noise_floor_db", noise_floor_db

			if (power_db > tb.squelch_threshold) and (freq >= tb.spec_min_freq) and (freq <= tb.spec_max_freq):     # and ((freq < tb.tx_freq_min) or (freq > tb.tx_freq_max)):
				print "DETECTION: ", datetime.now(), "center_freq", center_freq, "freq", freq, "power_db", power_db, "noise_floor_db", noise_floor_db
				# stop transmission hopping
				channel_grp_sel = range(tb.n_channel_grps)
				channel_grp_sel.remove(tb.curr_channel_grp)
				tb.set_channel_group(random.choice(channel_grp_sel))
				break
				# restart transmission hopping

if __name__ == '__main__':
	t = ThreadClass()
	t.start()

	tb = build_block()
	try:
		tb.start()
		main_loop(tb)

	except KeyboardInterrupt:
		pass

