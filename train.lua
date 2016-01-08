cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Video Captioning model')
cmd:text()
cmd:text('Options')

cmd:option('-model','meanpool','type of model to use? meanpool|frames|frames_cnn')

cmd:text()
cmd:option('-id', 1, 'an id identifying this run/job. used when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', -1, 'which gpu to use. -1 = use CPU')

local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)