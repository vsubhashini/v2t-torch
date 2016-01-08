local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)

  local info = path.join(opt.save_dir, 'info.json')
  local trainBatches = path.join(opt.save_dir, 'train.t7')
  local valBatches = path.join(opt.save_dir, 'val.t7')
  local valEvalBatches = path.join(opt.save_dir, 'valEval.t7')

  local prepro = false
  if not path.exists(info) then
    prepro = true
  else
    local curDataInfo = utils.read_json(info)
    if curDataInfo['batch_size'] ~= opt.batch_size or curDataInfo['max_seqlen'] ~= opt.max_seqlen then
      prepro = true
    end
  end

  if prepro then
    print('running one time pre-processing...')
    local labelData = {path.join(opt.label_dir, 'labels_train.txt'), path.join(opt.label_dir, 'labels_val.txt')}
    local videoData = {path.join(opt.video_dir, 'videos_train.txt'), path.join(opt.video_dir, 'videos_val.txt')}

    self:preprocess(labelData, videoData, opt.vocab_file, opt.save_dir, opt.batch_size, opt.max_seqlen)
  end

  print('loading preprocessed files...')
  local dataInfo = utils.read_json(info)

  -- load vocabulary
  self.word_to_ix = dataInfo['word_to_ix']
  self.ix_to_word = dataInfo['ix_to_word']
  self.vocabSize = 0
  for k,v in pairs(self.word_to_ix) do
    self.vocabSize = self.vocabSize + 1
  end 

  self.iter = {0, 0, 0}
  self.splitSizes = dataInfo['splitSizes']
  self.splits = {torch.load(trainBatches), torch.load(valBatches), torch.load(valEvalBatches)}
end

function DataLoader:getBatch(split_ix)
  self.iter[split_ix] = self.iter[split_ix]+1
  if self.iter[split_ix] > self.splitSizes[split_ix] then
    self.iter[split_ix] = 1
  end

  local batchId = self.iter[split_ix]
  return self.splits[split_ix][batchId][1], self.splits[split_ix][batchId][2], self.splits[split_ix][batchId][3]
end

function DataLoader:getVocabSize()
  return self.vocabSize
end

function DataLoader:getVocab()
  return self.ix_to_word
end

function DataLoader:splitSize(split_ix)
  return #self.splits[split_ix]
end

function DataLoader:resetIterator(split_ix)
  self.iter[split_ix] = 0
end

function DataLoader:preprocess(label_files, video_files, vocab_file, save_dir, batch_size, max_seqlen)
  local rawdata, rawdata2 -- two streams for reading raw data

  local out = {} -- info json struct 
  out['batch_size'] = batch_size
  out['max_seqlen'] = max_seqlen

  print('creating vocabulary mapping...')
  local word_to_ix = {}
  local ix_to_word = {}

  print(vocab_file)
  local f = io.open(vocab_file)
  rawdata = f:read()
  local vocabSize = 0
  repeat 
    vocabSize = vocabSize + 1
    word_to_ix[rawdata] = vocabSize
    ix_to_word[vocabSize] = rawdata
    rawdata = f:read()
  until not rawdata
  f:close()

  out['word_to_ix'] = word_to_ix
  out['ix_to_word'] = ix_to_word

  local splits = {'train', 'val'}
  local splitSizes = {}

  for splitIx=1, #splits do

    print('processing labels and videos for ' .. splits[splitIx] .. '...')
    local vidlabelPairs = {} -- table that stores video-caption pairs 
    local ids = {} -- store video ids that correspond to entries in vidlabelPairs 
    local labelFile = io.open(label_files[splitIx])
    local vidFile = io.open(video_files[splitIx])
    local curVidId = ''
    local curVidTensor
    
    rawdata = labelFile:read()
    repeat
      local cap = utils.split(rawdata, '%s')
      local caplen = #cap

      if caplen > max_seqlen - 1 then -- -1 because frame + word_seq must equal max_seqlen
        caplen = max_seqlen - 1
      end

      local capTensor = torch.Tensor(caplen) -- #cap because cap[1] is the video id but add END token to tensor
      for ix=2, caplen do
        capTensor[ix - 1] = word_to_ix[cap[ix]]
      end
      capTensor[caplen] = vocabSize + 1 -- END TOKEN

      if cap[1] ~= curVidId then
        rawdata2 = vidFile:read()
        local convnetFeats = utils.split(rawdata2, ',')

        assert(#convnetFeats == 4096)

        curVidTensor = torch.Tensor(#convnetFeats)
        for ix=1,#convnetFeats do
          curVidTensor[ix] = convnetFeats[ix]
        end
        curVidId = cap[1]
      end
      table.insert(vidlabelPairs, {curVidTensor, capTensor})
      table.insert(ids, curVidId)

      rawdata = labelFile:read()
    until not rawdata

    -- create batches
    print('creating batches for ' .. splits[splitIx] .. '...')
    local batches = {}
    -- make even bathes
    if #vidlabelPairs % batch_size ~=0 then 
      repeat
        table.remove(vidlabelPairs, #vidlabelPairs)
      until #vidlabelPairs % batch_size == 0
    end

    local splitSize = #vidlabelPairs/batch_size
    table.insert(splitSizes, splitSize)
    local rand = torch.randperm(#vidlabelPairs)
    for b=1, splitSize do
      -- find len of longest comment in batch
      longestCap = 0
      for b_ix=1,batch_size do
        local ix = rand[(b-1)*batch_size + b_ix]
        if vidlabelPairs[ix][2]:nElement() > longestCap then
          longestCap = vidlabelPairs[ix][2]:nElement()
        end
      end

      local batchVideos = torch.Tensor(batch_size, 4096):zero()
      local batchLabels = torch.Tensor(longestCap, batch_size):zero()
      local batchIds = {}
      for b_ix=1,batch_size do
        local ix = rand[(b-1)*batch_size + b_ix]
        local vid = vidlabelPairs[ix][1]
        local cap = vidlabelPairs[ix][2]
        batchVideos:select(1,b_ix):copy(vid)
        batchLabels:select(2,b_ix):sub(1,cap:nElement()):copy(cap)
        table.insert(batchIds, ids[ix])
      end
      table.insert(batches, {batchVideos, batchLabels, batchIds})
    end

    torch.save(path.join(save_dir, splits[splitIx] .. '.t7'), batches)
    -- generate an extra valEval file
    if splitIx == 2 then
      local valEval = {}
      local idDict = {}
      for s=1,#vidlabelPairs do
        if idDict[ids[s]] == nil then
          table.insert(valEval, {vidlabelPairs[s][1], vidlabelPairs[s][2], ids[s]})
          idDict[ids[s]] = true
        end
      end
      table.insert(splitSizes, #valEval)
      torch.save(path.join(save_dir, 'valEval.t7'), valEval)
    end
  end

  out['splitSizes'] = splitSizes
  utils.write_json(path.join(save_dir, 'info.json'), out)
end