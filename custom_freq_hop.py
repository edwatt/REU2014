#!/usr/bin/env python

"""
Custom freq hopping example - Hops 10 MHz at a time in a custom pattern back and forth increasing the signal amplitude on each loop
"""
from gnuradio import gr
from gnuradio import analog
from gnuradio import uhd
from time import sleep
from optparse import OptionParser
import sys
from gnuradio.eng_option import eng_option


MAX_RATE = 1000e6

class build_block(gr.top_block):
	def __init__(self):
		gr.top_block.__init__(self)

		args = "" #only supporting USB USRPs for now

		#find uhd devices

		d = uhd.find_devices(uhd.device_addr(args))
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
		self.u_tx = uhd.usrp_sink(device_addr=args, stream_args=stream_args)
		self.u_tx.set_samp_rate(MAX_RATE)

		#analog signal source - sig_source_c(sampling_freq,waveform, wave_freq, ampl, offset=0)
		self.tx_src0 = analog.sig_source_c(self.u_tx.get_samp_rate(), analog.GR_CONST_WAVE, 0, 1.0, 0)

		#check and output freq range, gain range, num_channels

		#gain range and max
		tx_gain_range = self.u_tx.get_gain_range()
		tx_gain_min = tx_gain_range.start()
		tx_gain_max = tx_gain_range.stop()

		#freq range
		tx_freq_range = self.u_tx.get_freq_range()
		tx_freq_low = tx_freq_range.start()
		tx_freq_high = tx_freq_range.stop()
		tx_freq_mid = (tx_freq_low + tx_freq_high) / 2.0

		#output info
		print "\nDevice Info"
		print "\n\tType: %s" % uhd_type

		print "\n\tMin Freq: %d MHz" % (tx_freq_low/1e6)
		print "\tMax Freq: %d MHz" % (tx_freq_high/1e6)
		print "\tMid Freq: %d MHz" % (tx_freq_mid/1e6)

		print "\n\tMin Gain: %d dB" % tx_gain_min
		print "\tMax Gain: %d dB" % tx_gain_max

		#set initial parameters 

		for i in xrange(tx_nchan):
			self.u_tx.set_center_freq(tx_freq_mid + i*1e6, i)
			self.u_tx.set_gain(tx_gain_max, i)

		#connect blocks

		self.connect(self.tx_src0, self.u_tx)

def main():

	parser = OptionParser (option_class=eng_option)
	parser.add_option ("-o", "--offset", type="eng_float", default=0,
					   help="set offset from center frequency", metavar="OFFSET")

	(options, args) = parser.parse_args()

	if len(args) != 0:
		parser.print_help()
		sys.exit(1)

	if options.offset < 1e6:
		options.offset *= 1e6



	try:
		tb = build_block()
		tb.start()

		if tb.u_tx is not None:

			print "Application will hop through a 50 MHz band in the center of the operating frequency 5 times increasing the signal amplitude each time"
			raw_input("Press Enter to begin transmission & Ctrl-C to exit\n")
			
			start = tb.u_tx.get_freq_range().start()
			stop = tb.u_tx.get_freq_range().stop()

			freq_hops = 5
		
			print "\nTransmit Frequencies:"

			channel = 0 #default to first channel
			trans_amp = 0.2

			while trans_amp <= 1.0:

				tb.u_tx.set_gain(trans_amp, channel)

				for i in xrange(freq_hops):
					trans_freq = ( start + stop ) / 2 + options.offset - freq_hops / 2.0 + i * 10e6
					tb.u_tx.set_center_freq(trans_freq,channel)
					print "\t%d MHz @ %f" % ((trans_freq/1e6), trans_amp)
					sleep(.5)

				trans_amp += 0.2
				if trans_amp > 1.0: break

				tb.u_tx.set_gain(trans_amp, channel)

				for i in xrange(freq_hops - 1, -1, -1):
					trans_freq = ( start + stop ) / 2 + options.offset - freq_hops / 2.0 + i * 10e6
					tb.u_tx.set_center_freq(trans_freq,channel)
					print "\t%d MHz @ %f" % ((trans_freq/1e6), trans_amp)
					sleep(.5)

				trans_amp += 0.2

		
		print "\nTest Over"

		tb.stop()
	except KeyboardInterrupt:
		pass


if __name__ == '__main__':
	main()
