<template>
  <div class="slidev-layout center flex flex-col h-full">
    <slot name="default">
      <h1>{{ $slidev.configs.title }}</h1>
    </slot>
    <div class="flex flex-1 items-center justify-center">
      <slot name="content" />
    </div>
  </div>
</template>
