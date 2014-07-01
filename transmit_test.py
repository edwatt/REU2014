#!/usr/bin/env python

"""
Retrieve operating parameters of connected USRP and loop through the operating spectrum trasmitting a constant wave signal
"""
from gnuradio import gr, eng_notation
from gnuradio import analog
from gnuradio import uhd
from gnuradio.eng_option import eng_option
from optparse import OptionParser
from time import sleep

MAX_RATE = 1000e6

class build_block(gr.top_block):
	def __init__(self):
		gr.top_block.__init__(self)
		
		usage = "usage: %prog [options]"
		parser = OptionParser(option_class=eng_option, usage=usage)
		parser.add_option("-f", "--tx-freq", type="eng_float", default=None,
						  metavar="Hz", help="Transmit frequency [default=center_frequency]")

		(options, args) = parser.parse_args()

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

		if options.tx_freq is None:
			self.tx_freq = tx_freq_mid
		else:
			if options.tx_freq < 1e6:
				options.tx_freq *= 1e6
			self.tx_freq = options.tx_freq

		#output info
		print "\nDevice Info -\tType: %s\tFreq(MHz): (%d,%d,%d)\tGain(dB): (%f,%f)\n" % (uhd_type, (tx_freq_low/1e6), (tx_freq_mid/1e6), (tx_freq_high/1e6), tx_gain_min, tx_gain_max)		

		#set initial parameters 
		self.u_tx.set_center_freq(self.tx_freq)
		self.u_tx.set_gain(tx_gain_max)

		#connect blocks

		self.connect(self.tx_src0, self.u_tx)

def main():

	try:
		tb = build_block()
		tb.start()

		if tb.u_tx is not None:

			print "Transmission will trasmit at a single frequency"
			print "Frequency: %d MHz" % (tb.tx_freq / 1e6)
			print "Transmission ON"
 
			while(1):
				raw_input("Press Enter to toggle transmission & Ctrl-C to exit\n")
				if tb.tx_src0.amplitude() == 1.0:
					tb.tx_src0.set_amplitude(0.0)
					print "Transmission OFF"
				else:
					tb.tx_src0.set_amplitude(1.0)
					print "Transmission ON"
		
		print "\nTest Over"

		tb.stop()
	except KeyboardInterrupt:
		print "\nTest Over"


if __name__ == '__main__':
	main()
