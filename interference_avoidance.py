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
from optparse import OptionParser
import sys
import math
import struct
from gnuradio.eng_option import eng_option
import threading

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
		parser = OptionsParser(option=eng_option, usage=usage)
		parser.add_option("-a", "--args", type="string", default=""
						  help="UHD device address args [default=%default]")
		parser.add_option("", "--spec", type="string", default=None,
						  help="Subdevice of UHD device where appropriate")
		parser.add_option("-R", "--rx-antenna", type="string", default=None,
						  help="select RX antenna where appropriate")
		parser.add_option("-T", "--tx-antenna", type="string", default=None,
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

		(options, args) = parser.parse_args()
		if len(args) != 2:
			parser.print_help()
			sys.exit(1)

		self.channel_bandwidth = options.channel_bandwidth

		self.min_freq = eng_notation.str_to_num(args[0])
		self.max_freq = eng_notation.str_to_num(args[1])

		if self.min_freq > self.max_freq:
			#swap them
			self.min_freq, self.max_freq = self.max_freq, self.min_freq

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
		if options.antenna:
			self.u_rx.set_antenna(options.rx_antenna, 0)

		self.u_rx.set_samp_rate(options.samp_rate)
		self.usrp_rate = usrp_rate = self.u_rx.get_samp_rate()

		self.lo_offset = options.lo_offset

		if options.fft_size is None:
			self.fft_size = int(self.usrp_rate/self.channel_bandwidth)
		else
			self.fft_size = options.fft_size

		self.squelch_threshold = options.squelch_threshold

		s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, self.fft_size)

		mywindow=filter.window.blackmanharris(self.fft_size)
		ffter = fft.fft_vcc(self.fft_size, True, mywindow, True)
		power = 0

		for tap in mywindow:
			power += tap*tap

		c2mag = blocks.complex_to_mag_squared(self.fft_size)

		self.freq_step = self.nearest_freq((0.75 * self.usrp_rate), self.channel_bandwidth)
		self.min_center_freq = self.min_freq + (self.freq_step / 2)
		nsteps = math.ceil((self.max_freq - self.min_freq) / self.freq_step)
		self.max_center_freq = self.min_center_freq + (nsteps * self.freq_step)

		self.next_freq = self.min_center_freq

		tune_delay = max(0, int(round(options.tune_delay * usrp_rate / self.fft_size))) # in fft_frames
		dwell_delay = max(1, int(round(options.dwell_delay * usrp_rate / self.fft_size))) # in fft frames

		self.msgq = gr.msg_queue(1)
		self._tune_callback = tune(self)   #hang on to this to keep it from being GC'd
	
		stats = blocks.bin_statistics_f(self.fft_size, self.msgq, self._tune_callback, tune_delay, dwell_delay)

		self.connect(self.u_rx, s2v, ffter, c2mag, stats)

		if options.gain is None:
			# if no gain was specified use the midpoint in dB
			g = self.u_rx.get_gain_range()
			options.gain = float(g.start()+g.stop())/2.0

		self.set_gain(options.gain)
		print "gain =", options.gain

	def set_next_freq(self):
		target_freq = self.next_freq
		self.next_freq = self.next_freq + self.freq_step
		if self.next_freq >= self.max_center_freq:
			self.next_freq = self.min_center_freq

		if not self.set_freq(target_freq):
			print "Failed to set frequency to", target_freq
			sys.exit(1)

		return target_freq

	def set_freq(self, target_freq):
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

	def nearest_freq(self, freq, channel_bandwidth):
		freq = round(freq / channel_bandwidth, 0) * channel_bandwidth
		return freq		

def main_loop(tb):

	def bin_freq(i_bin, center_freq):
		freq = center_freq - (tb.usrp_rate / 2) + (tb.channel_bandwidth * i_bin)
		return freq

	bin_start = int(tb.fft_size * ((1 - 0.75) / 2))
	bin_stop = int(tb.fft_size - bin_start)

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
			freq.bin_freq(i_bin, center_freq)
			noise_floor_db = 10*math.log10(min(m.data)/tb.usrp_rate)
			power_db = 10*math.log10(m.data[i_bin]/tb.usrp_rate) - noise_floor_db

			if (power_db > tb.squelch_threshold) and (freq >= tb.min_freq) and (freq <= tb.max_freq)
				print datetime.now(), "center_freq", center_freq, "freq", freq, "power_db", power_db, "noise_floor_db", noise_floor_db


if __name__ == '__main__':
	t = ThreadClass()
    t.start()

    tb = build_block()
    try:
        tb.start()
        main_loop(tb)

    except KeyboardInterrupt:
        pass

