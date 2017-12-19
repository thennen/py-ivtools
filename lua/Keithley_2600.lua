function SweepVList(sweepList, rangeI, limitI, nplc, delay)
    reset()

    -- Configure the SMU
    smua.reset()
    smua.source.func			= smua.OUTPUT_DCVOLTS
    smua.source.limiti			= limitI
    smua.measure.nplc			= nplc
    --smua.measure.delay		= smua.DELAY_AUTO
    smua.measure.delay = delay
    smua.measure.rangei = rangeI
    --smua.measure.rangev = 0

    -- Prepare the Reading Buffers
    smua.nvbuffer1.clear()
    smua.nvbuffer1.collecttimestamps	= 1
    --smua.nvbuffer1.collectsourcevalues  = 1
    smua.nvbuffer2.clear()
    smua.nvbuffer2.collecttimestamps	= 1
    smua.nvbuffer2.collectsourcevalues  = 1

    -- Configure SMU Trigger Model for Sweep
    smua.trigger.source.listv(sweepList)
    smua.trigger.source.limiti			= limitI
    smua.trigger.measure.action			= smua.ENABLE
    smua.trigger.measure.iv(smua.nvbuffer1, smua.nvbuffer2)
    smua.trigger.endpulse.action		= smua.SOURCE_HOLD
    -- By setting the endsweep action to SOURCE_IDLE, the output will return
    -- to the bias level at the end of the sweep.
    smua.trigger.endsweep.action		= smua.SOURCE_IDLE
    numPoints = table.getn(sweepList)
    smua.trigger.count					= numPoints
    smua.trigger.source.action			= smua.ENABLE
    -- Ready to begin the test

    smua.source.output					= smua.OUTPUT_ON
    -- Start the trigger model execution
    smua.trigger.initiate()
end
