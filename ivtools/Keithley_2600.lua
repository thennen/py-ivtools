function SweepVList(sweepList, rangeI, limitI, nplc, delay, rangeV)
    reset()

    -- Configure the SMU
    smua.reset()
    smua.source.func		= smua.OUTPUT_DCVOLTS
    smua.measure.nplc	  = nplc
    --smua.measure.delay		= smua.DELAY_AUTO
    smua.measure.delay  = delay
    if limitI ~= 0 then
      smua.source.limiti  = limitI
    end
    -- Autorange if you pass limitI = 0
    -- Keithley programmers guide 7-226
    -- "Explicitly setting a measure range will disable measure autoranging for that function."
    if rangeI ~= 0 then
      smua.measure.rangei = rangeI
    end

    -- In case you want to also fix the measurement range of the source
    -- Autoranging can sometimes cause problems
    if rangeV ~=0 then
      -- Not convinced that this really works ..
      smua.source.autorangev = smua.AUTORANGE_OFF
      -- Need to specify or no data comes back..
      smua.source.rangev = rangeV
      --smua.measure.rangev = rangeV
    end

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


function SweepIList(sweepList, rangeV, limitV, nplc, delay, rangeI)
    reset()

    -- Configure the SMU
    smua.reset()
    smua.source.func			= smua.OUTPUT_DCAMPS
    smua.measure.nplc			= nplc
    --smua.measure.delay		= smua.DELAY_AUTO
    smua.measure.delay = delay

    if limitV ~= 0 then
      smua.source.limitv		= limitV
    end

    -- Keithley programmers guide 7-226
    -- "Explicitly setting a measure range will disable measure autoranging for that function."
    if rangeV ~=0 then
      smua.measure.rangev = rangeV
    end

    -- In case you want to also fix the measurement range of the source
    -- Autoranging can always cause problems
    if rangeI ~=0 then
      smua.source.autorangei = smua.AUTORANGE_OFF
      -- Need to specify or no data comes back..
      smua.source.rangei = rangeI
    end

    -- Prepare the Reading Buffers
    smua.nvbuffer1.clear()
    smua.nvbuffer1.collecttimestamps	= 1
    smua.nvbuffer1.collectsourcevalues  = 1
    smua.nvbuffer2.clear()
    smua.nvbuffer2.collecttimestamps	= 1
    --smua.nvbuffer2.collectsourcevalues  = 1

    -- Configure SMU Trigger Model for Sweep
    smua.trigger.source.listi(sweepList)
    smua.trigger.source.limitv			= limitV
    smua.trigger.measure.action			= smua.ENABLE
    smua.trigger.measure.iv(smua.nvbuffer1, smua.nvbuffer2)
    -- smua.trigger.endpulse.action		= smua.SOURCE_HOLD
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

-- Stefan Wiefels wrote these:

function SweepVList2CH(sweepList1, VB, rangeI, limitI, nplc, delay)
    reset()

    -- Configure the SMU
    smua.reset()
    smua.source.func            = smua.OUTPUT_DCVOLTS
    smua.source.limiti          = limitI
    smua.measure.nplc           = nplc
    --smua.measure.delay        = smua.DELAY_AUTO
    smua.measure.delay = delay
    smua.measure.rangei = rangeI
    --smua.measure.rangev = 0

    -- Prepare the Reading Buffers
    smua.nvbuffer1.clear()
    smua.nvbuffer1.collecttimestamps    = 1
    --smua.nvbuffer1.collectsourcevalues  = 1
    smua.nvbuffer2.clear()
    smua.nvbuffer2.collecttimestamps    = 1
    smua.nvbuffer2.collectsourcevalues  = 1
    --smub.nvbuffer.clear()
    --smub.nvbuffer.count

    -- Configure SMU Trigger Model for Sweep
    smua.trigger.source.listv(sweepList1)
    smua.trigger.source.limiti          = limitI
    smua.trigger.measure.action         = smua.ENABLE
    smua.trigger.measure.iv(smua.nvbuffer1, smua.nvbuffer2)
    smua.trigger.endpulse.action        = smua.SOURCE_HOLD
    -- By setting the endsweep action to SOURCE_IDLE, the output will return
    -- to the bias level at the end of the sweep.
    smua.trigger.endsweep.action        = smua.SOURCE_IDLE
    numPoints = table.getn(sweepList1)
    smua.trigger.count                  = numPoints
    smua.trigger.source.action          = smua.ENABLE

    -- Ready to begin the test
    smua.source.output                  = smua.OUTPUT_ON
    smub.source.output                  = smub.OUTPUT_ON
    -- Start the trigger model execution


    -- Set ChB
    smub.reset()
    smub.source.func            = smub.OUTPUT_DCVOLTS
    smub.source.limiti          = limitI
    smub.measure.rangei         = limitI
    smub.source.levelv = VB
    --smub.measure.count = table.getn(sweepList1)
    --smub.measure.overlappedi(smub.nvbuffer)
    smua.trigger.initiate()
end

function constantVMeasI(sourceVA, sourceVB, points, interval, rangeI, limitI, nplc)

    reset()

    -- Configure the SMU
    smua.reset()
    smua.source.func            = smua.OUTPUT_DCVOLTS
    smua.source.limiti          = limitI
    smua.measure.nplc           = nplc
    -- Autorange option added by Moritz
    if rangeI == 0 then
      smua.measure.autorangei = smua.AUTORANGE_ON
    else
      smua.measure.rangei = rangeI
    end

    -- Prepare the Reading Buffers
    smua.nvbuffer1.clear()
    smua.nvbuffer1.collecttimestamps    = 1
    -- What are the source values for voltage if we are sourcing current?
    smua.nvbuffer1.collectsourcevalues  = 1
    smua.nvbuffer2.clear()
    smua.nvbuffer2.collecttimestamps    = 1
    smua.nvbuffer2.collectsourcevalues  = 1

    smua.source.levelv          = sourceVA
    smua.measure.count          = points
    smua.measure.interval       = interval

    -- Set ChB
    smub.reset()
    smub.source.func            = smua.OUTPUT_DCVOLTS
    smub.source.limiti          = limitI
    smub.measure.rangei         = limitI
    smub.source.levelv          = sourceVB

    -- Ready to begin the test
    smua.source.output                  = smua.OUTPUT_ON
    smub.source.output                  = smub.OUTPUT_ON
    -- Start the trigger model execution

    smua.measure.overlappediv(smua.nvbuffer1, smua.nvbuffer2)

end
